"""Detection rule: despesa desproporcional — a normally-cheap item billed at a value far above
what the same item usually costs in comparable campaigns (same year and office).

Indício, não prova: the TSE publishes only the total amount, never the quantity bought."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from statistics import median
from typing import NamedTuple

from ..log import RowCounter, get_logger, step
from ..util import git_commit, now_utc
from .common import reset_rule_signals

log = get_logger("elosys.rules.disproportionate_expense")

RULE_NAME = "disproportionate_expense"
RULE_VERSION = "3.0"

CATEGORIES: tuple[tuple[str, ...], ...] = (
    ("CANETA",),
    ("LAPIS", "LÁPIS"),
    ("LAPISEIRA",),
    ("BORRACHA",),
    ("APONTADOR",),
    ("ADESIVO",),
    ("CRACHA", "CRACHÁ"),
    ("ETIQUETA",),
    ("CLIPS",),
    ("GRAMPO",),
    ("GRAMPEADOR",),
    ("REGUA", "RÉGUA"),
    ("BLOCO DE ANOTA",),
    ("ENVELOPE",),
    ("MARCADOR DE TEXTO",),
    ("PRANCHETA",),
    ("PERFURADOR",),
    ("ELASTICO", "ELÁSTICO"),
)
KEYWORDS = tuple(spelling for cat in CATEGORIES for spelling in cat)
CANONICAL_OF = {spelling: cat[0] for cat in CATEGORIES for spelling in cat}

MIN_SAMPLE_FOR_STATS = 20
MEDIAN_MULTIPLIER_MEDIUM = 15
MEDIAN_MULTIPLIER_HIGH = 30
MIN_FLOOR_CENTS = 100_000
FALLBACK_MEDIUM_FLOOR_CENTS = 500_000
FALLBACK_HIGH_FLOOR_CENTS = 5_000_000

PARAMS = {
    "categories": [list(cat) for cat in CATEGORIES],
    "min_sample_for_stats": MIN_SAMPLE_FOR_STATS,
    "median_multiplier_medium": MEDIAN_MULTIPLIER_MEDIUM,
    "median_multiplier_high": MEDIAN_MULTIPLIER_HIGH,
    "min_floor_cents": MIN_FLOOR_CENTS,
    "fallback_medium_floor_cents": FALLBACK_MEDIUM_FLOOR_CENTS,
    "fallback_high_floor_cents": FALLBACK_HIGH_FLOOR_CENTS,
    "reference_scopes": ["category+year+office", "category+year"],
    "multi_item_policy": "excluded from medians; fixed floors",
}


class _Reference(NamedTuple):
    scope: str
    n: int
    median_cents: float
    medium_floor: int
    high_floor: int


def _categories_in(description: str) -> set[str]:
    """Categories named in a description. Longer spellings are matched first and removed from
    the text, so "LAPISEIRA" is not also counted as "LAPIS"."""
    found: set[str] = set()
    remaining = description
    for spelling in sorted(CANONICAL_OF, key=len, reverse=True):
        if spelling in remaining:
            found.add(CANONICAL_OF[spelling])
            remaining = remaining.replace(spelling, " ")
    return found


def _scope_label(key: tuple) -> str:
    _, year, office = key
    return f"{office} {year}" if office is not None else f"eleição de {year}"


def _build_references(single_item_rows: list[tuple[sqlite3.Row, str]]) -> dict[tuple, _Reference]:
    """Median per (category, year, office) and per (category, year), only where the sample is
    big enough. Municipal and general elections have very different price levels, so one
    pooled median would make every state/federal expense look inflated."""
    amounts: dict[tuple, list[int]] = defaultdict(list)
    for r, category in single_item_rows:
        amount = r["amount_cents"] or 0
        amounts[(category, r["year"], None)].append(amount)
        if r["office"] is not None:
            amounts[(category, r["year"], r["office"])].append(amount)

    references = {}
    for key, values in amounts.items():
        if len(values) < MIN_SAMPLE_FOR_STATS:
            continue
        med = median(values)
        references[key] = _Reference(
            scope=_scope_label(key), n=len(values), median_cents=med,
            medium_floor=max(MIN_FLOOR_CENTS, round(med * MEDIAN_MULTIPLIER_MEDIUM)),
            high_floor=max(MIN_FLOOR_CENTS, round(med * MEDIAN_MULTIPLIER_HIGH)),
        )
    return references


def _reference_for(references: dict[tuple, _Reference], category: str, year: int,
                   office: str | None) -> _Reference | None:
    return references.get((category, year, office)) or references.get((category, year, None))


def _assess(amount: int, categories: set[str], year: int, office: str | None,
            references: dict[tuple, _Reference]) -> tuple[str, str] | None:
    """(severity, basis text) when the amount is flagged, else None."""
    category = next(iter(categories))
    reference = _reference_for(references, category, year, office) if len(categories) == 1 else None

    if reference is None:
        if amount < FALLBACK_MEDIUM_FLOOR_CENTS:
            return None
        severity = "high" if amount >= FALLBACK_HIGH_FLOOR_CENTS else "medium"
        if len(categories) > 1:
            items = ", ".join(sorted(c.lower() for c in categories))
            return severity, (f"descrição com vários itens baratos ({items}), "
                              "sem comparação com a mediana — piso fixo aplicado")
        return severity, (f'categoria "{category.lower()}" com poucas despesas '
                          f"comparáveis em {year} — piso fixo aplicado")

    if amount < reference.medium_floor:
        return None
    severity = "high" if amount >= reference.high_floor else "medium"
    if not reference.median_cents:
        return severity, "categoria sem mediana estável"
    return severity, (
        f"{amount / reference.median_cents:.0f}x a mediana de R$ {reference.median_cents / 100:,.2f} "
        f'pra "{category.lower()}" em {reference.scope} ({reference.n:,} despesas comparáveis)'
    )


def run(con: sqlite3.Connection) -> dict:
    with step(log, "reset disproportionate_expense signals"):
        reset_rule_signals(con, RULE_NAME)

    where_keywords = " OR ".join("ce.description LIKE ?" for _ in KEYWORDS)
    like_params = [f"%{k}%" for k in KEYWORDS]
    with step(log, "scan campaign_expense for cheap-item categories"):
        rows = con.execute(
            "SELECT ce.id, ce.amount_cents, ce.description, ce.year, ce.cnpj, "
            "       ce.supplier_name, ce.supplier_company_id, ce.supplier_person_id, "
            "       co.person_id AS candidate_person_id, co.office AS office "
            "FROM campaign_expense ce "
            "LEFT JOIN campaign_org co ON co.id = ce.campaign_org_id "
            f"WHERE ({where_keywords})",  # noqa: S608
            like_params,
        ).fetchall()

    classified = [(r, cats) for r in rows if (cats := _categories_in(r["description"]))]
    single_item = [(r, next(iter(cats))) for r, cats in classified if len(cats) == 1]
    references = _build_references(single_item)
    log.info("  %s single-item expenses, %s multi-item (fixed floors), %s reference medians",
             f"{len(single_item):,}", f"{len(classified) - len(single_item):,}", f"{len(references):,}")

    cur = con.execute(
        "INSERT INTO rule_run (rule, rule_version, code_commit, params, run_at, rows_generated) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (RULE_NAME, RULE_VERSION, git_commit(), json.dumps(PARAMS, ensure_ascii=False), now_utc()),
    )
    rule_run_id = int(cur.lastrowid)

    rc = RowCounter(log, "candidate expenses scanned", every=50_000)
    generated = 0
    for r, cats in classified:
        rc.tick()
        amount = r["amount_cents"] or 0
        assessment = _assess(amount, cats, r["year"], r["office"], references)
        if assessment is None:
            continue
        severity, basis = assessment

        explanation = (
            f"Despesa de campanha de R$ {amount / 100:,.2f} descrita como "
            f'"{r["description"]}" — {basis}. '
            "Pode ser lote com itens não detalhados na descrição, compra em grande "
            "volume, ou erro de digitação no valor; não é, por si só, indício de "
            "irregularidade."
        )
        sig_cur = con.execute(
            "INSERT INTO signal (rule_run_id, type, severity, explanation, amount_cents) VALUES (?, ?, ?, ?, ?)",
            (rule_run_id, "cheap_item_high_value", severity, explanation, amount),
        )
        signal_id = int(sig_cur.lastrowid)

        actors = []
        if r["candidate_person_id"] is not None:
            actors.append(("person", r["candidate_person_id"], "candidate"))
        if r["supplier_person_id"] is not None:
            actors.append(("person", r["supplier_person_id"], "supplier"))
        if r["supplier_company_id"] is not None:
            actors.append(("company", r["supplier_company_id"], "supplier"))
        con.executemany(
            "INSERT OR IGNORE INTO signal_actor (signal_id, type, actor_id, role) VALUES (?, ?, ?, ?)",
            [(signal_id, t, aid, role) for t, aid, role in actors],
        )
        con.execute(
            "INSERT INTO signal_evidence (signal_id, table_name, record_id) VALUES (?, 'campaign_expense', ?)",
            (signal_id, r["id"]),
        )
        generated += 1
    rc.done()

    con.execute("UPDATE rule_run SET rows_generated = ? WHERE id = ?", (generated, rule_run_id))
    con.commit()

    by_severity = dict(con.execute(
        "SELECT severity, count(*) FROM signal WHERE rule_run_id = ? GROUP BY severity",
        (rule_run_id,),
    ).fetchall())
    log.info("done: %s sinais gerados (rule_run %d) — %s",
             f"{generated:,}", rule_run_id, by_severity)
    return {"rule_run_id": rule_run_id, "signals": generated, "by_severity": by_severity}
