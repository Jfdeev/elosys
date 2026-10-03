"""Detection rule: empresa de fachada — a supplier whose Receita Federal registry data is hard to
square with the money it received from campaigns.

Three signals, from strongest to weakest:
  * paid_before_company_opened  (high)   expense dated before the company's start of activity
  * paid_after_company_inactive (high)   expense dated after the company became baixada/inapta/suspensa
  * new_company_low_capital_high_value (medium)  opened <= 180 days before its first payment,
                                         received >= R$ 50k in total, with share capital <= R$ 10k

Indício, não prova: registry dates can be wrong, a CNPJ can be reused, and many legitimate
companies are opened right before an election or run on tiny capital. Only suppliers already in
company_registry are evaluated (that table is filled incrementally by `elosys receita-cnpj`).
Other campaigns' CNPJs (transfers between campaigns) and party committees are left out.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import date
from itertools import groupby
from typing import NamedTuple

from ..log import RowCounter, get_logger, step
from ..util import git_commit, now_utc
from .common import reset_rule_signals

log = get_logger("elosys.rules.shell_supplier")

RULE_NAME = "shell_supplier"
RULE_VERSION = "1.0"

TYPE_BEFORE_OPENED = "paid_before_company_opened"
TYPE_AFTER_INACTIVE = "paid_after_company_inactive"
TYPE_NEW_LOW_CAPITAL = "new_company_low_capital_high_value"

INACTIVE_STATUSES = frozenset({"BAIXADA", "INAPTA", "SUSPENSA", "NULA"})
NEW_COMPANY_MAX_DAYS = 180
HIGH_VALUE_CENTS = 5_000_000
LOW_CAPITAL_CENTS = 1_000_000
MAX_EVIDENCE_ROWS = 20  # per signal: the largest expenses; the explanation carries the full count

PARAMS = {
    "inactive_statuses": sorted(INACTIVE_STATUSES),
    "new_company_max_days": NEW_COMPANY_MAX_DAYS,
    "high_value_cents": HIGH_VALUE_CENTS,
    "low_capital_cents": LOW_CAPITAL_CENTS,
    "max_evidence_rows": MAX_EVIDENCE_ROWS,
    "date_field": "campaign_expense.expense_date",
    "excluded": ["suppliers that are campaign CNPJs", "legal nature: partido político"],
}

_CAVEAT = "Pode ser erro de cadastro ou de data; não é, por si só, indício de irregularidade."

_EXPENSES_OF_REGISTERED_SUPPLIERS = """
SELECT ce.id, ce.amount_cents, ce.expense_date, ce.supplier_company_id,
       ce.supplier_cpf_cnpj, ce.supplier_name,
       cr.legal_name, cr.opened_at, cr.registry_status, cr.registry_status_date,
       cr.share_capital_cents,
       co.person_id AS candidate_person_id
FROM campaign_expense ce
JOIN company_registry cr ON cr.company_id = ce.supplier_company_id
LEFT JOIN campaign_org co ON co.id = ce.campaign_org_id
WHERE ce.expense_date IS NOT NULL
  AND cr.opened_at IS NOT NULL
  AND coalesce(cr.legal_nature, '') NOT LIKE '%Partido Pol%'
  AND NOT EXISTS (SELECT 1 FROM campaign_org other WHERE other.cnpj = ce.supplier_cpf_cnpj)
ORDER BY ce.supplier_company_id
"""


class Finding(NamedTuple):
    type: str
    severity: str
    amount_cents: int
    explanation: str
    company_id: int
    evidence_ids: list[int]
    candidate_person_ids: list[int]


def _days_between(earlier: str, later: str) -> int:
    return (date.fromisoformat(later[:10]) - date.fromisoformat(earlier[:10])).days


def _brl(cents: int) -> str:
    return f"R$ {cents / 100:,.2f}"


def _finding(type_: str, severity: str, rows: list[Mapping], explanation: str) -> Finding:
    biggest = sorted(rows, key=lambda r: r["amount_cents"] or 0, reverse=True)[:MAX_EVIDENCE_ROWS]
    candidates = sorted({r["candidate_person_id"] for r in rows if r["candidate_person_id"] is not None})
    return Finding(
        type=type_, severity=severity,
        amount_cents=sum(r["amount_cents"] or 0 for r in rows),
        explanation=explanation,
        company_id=rows[0]["supplier_company_id"],
        evidence_ids=[r["id"] for r in biggest],
        candidate_person_ids=candidates,
    )


def _company_label(row: Mapping) -> str:
    name = row["legal_name"] or row["supplier_name"] or "sem nome"
    return f"{name} (CNPJ {row['supplier_cpf_cnpj']})"


def findings_for_company(rows: list[Mapping]) -> list[Finding]:
    """Findings for one supplier, given every expense paid to it (all rows share its registry data)."""
    first = rows[0]
    who = _company_label(first)
    opened = first["opened_at"]
    findings: list[Finding] = []

    before = [r for r in rows if r["expense_date"] < opened]
    if before:
        earliest = min(r["expense_date"] for r in before)
        findings.append(_finding(
            TYPE_BEFORE_OPENED, "high", before,
            f"Empresa {who} tem início de atividade em {opened[:10]} na Receita Federal, mas "
            f"{len(before)} despesa(s) de campanha, somando {_brl(sum(r['amount_cents'] or 0 for r in before))}, "
            f"foram registradas antes dessa data (a primeira em {earliest[:10]}). {_CAVEAT}",
        ))

    status_date = first["registry_status_date"]
    if first["registry_status"] in INACTIVE_STATUSES and status_date:
        after = [r for r in rows if r["expense_date"] > status_date]
        if after:
            latest = max(r["expense_date"] for r in after)
            findings.append(_finding(
                TYPE_AFTER_INACTIVE, "high", after,
                f"Empresa {who} consta como {first['registry_status']} desde {status_date[:10]} na "
                f"Receita Federal, mas {len(after)} despesa(s) de campanha, somando "
                f"{_brl(sum(r['amount_cents'] or 0 for r in after))}, foram registradas depois "
                f"disso (a última em {latest[:10]}). {_CAVEAT}",
            ))

    capital = first["share_capital_cents"]
    total = sum(r["amount_cents"] or 0 for r in rows)
    days_to_first_payment = _days_between(opened, min(r["expense_date"] for r in rows))
    if (capital is not None and capital <= LOW_CAPITAL_CENTS and total >= HIGH_VALUE_CENTS
            and 0 <= days_to_first_payment <= NEW_COMPANY_MAX_DAYS):
        findings.append(_finding(
            TYPE_NEW_LOW_CAPITAL, "medium", rows,
            f"Empresa {who} foi aberta em {opened[:10]}, recebeu o primeiro pagamento de campanha "
            f"{days_to_first_payment} dia(s) depois e somou {_brl(total)} em {len(rows)} despesa(s), "
            f"com capital social de {_brl(capital)}. Empresas novas e de pouco capital também "
            f"contratam legitimamente; vale conferir a capacidade operacional.",
        ))
    return findings


def collect_findings(expense_rows: Iterable[Mapping]) -> list[Finding]:
    """Rows must arrive ordered by supplier_company_id."""
    findings: list[Finding] = []
    for _, group in groupby(expense_rows, key=lambda r: r["supplier_company_id"]):
        findings.extend(findings_for_company(list(group)))
    return findings


def run(con: sqlite3.Connection) -> dict:
    with step(log, "reset shell_supplier signals"):
        reset_rule_signals(con, RULE_NAME)

    with step(log, "scan expenses paid to suppliers with registry data"):
        findings = collect_findings(con.execute(_EXPENSES_OF_REGISTERED_SUPPLIERS))
    log.info("  %s findings", f"{len(findings):,}")

    cur = con.execute(
        "INSERT INTO rule_run (rule, rule_version, code_commit, params, run_at, rows_generated) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (RULE_NAME, RULE_VERSION, git_commit(), json.dumps(PARAMS, ensure_ascii=False), now_utc()),
    )
    rule_run_id = int(cur.lastrowid)

    rc = RowCounter(log, "signals written", every=500)
    for f in findings:
        rc.tick()
        signal_id = int(con.execute(
            "INSERT INTO signal (rule_run_id, type, severity, explanation, amount_cents) VALUES (?, ?, ?, ?, ?)",
            (rule_run_id, f.type, f.severity, f.explanation, f.amount_cents),
        ).lastrowid)
        actors = [("company", f.company_id, "supplier")]
        actors += [("person", pid, "candidate") for pid in f.candidate_person_ids]
        con.executemany(
            "INSERT OR IGNORE INTO signal_actor (signal_id, type, actor_id, role) VALUES (?, ?, ?, ?)",
            [(signal_id, t, aid, role) for t, aid, role in actors],
        )
        con.executemany(
            "INSERT INTO signal_evidence (signal_id, table_name, record_id) VALUES (?, 'campaign_expense', ?)",
            [(signal_id, eid) for eid in f.evidence_ids],
        )
    rc.done()

    con.execute("UPDATE rule_run SET rows_generated = ? WHERE id = ?", (len(findings), rule_run_id))
    con.commit()

    by_type = dict(con.execute(
        "SELECT type, count(*) FROM signal WHERE rule_run_id = ? GROUP BY type", (rule_run_id,),
    ).fetchall())
    by_severity = dict(con.execute(
        "SELECT severity, count(*) FROM signal WHERE rule_run_id = ? GROUP BY severity", (rule_run_id,),
    ).fetchall())
    log.info("done: %s sinais (rule_run %d) — %s %s", f"{len(findings):,}", rule_run_id, by_type, by_severity)
    return {"rule_run_id": rule_run_id, "signals": len(findings), "by_type": by_type,
            "by_severity": by_severity}
