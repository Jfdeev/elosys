"""Detection rule: shell_supplier, against synthetic company_registry / campaign_expense rows."""

from __future__ import annotations

import itertools

import pytest

from elosys.db import connect, create_schema
from elosys.rules import shell_supplier as rule

T = "2026-01-01T00:00:00Z"
CAMPAIGN_CNPJ = "40430149000110"

_cnpjs = itertools.count(10_000_000_000_001)
_expense_ids = itertools.count(1)


@pytest.fixture
def world(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    con.execute("INSERT INTO people (cpf, cpf_trusted, canonical_name, created_at) "
                "VALUES ('11144477735', 1, 'JOAO DA SILVA', ?)", (T,))
    con.execute("INSERT INTO source (name, agency, type, base_url, created_at) "
                "VALUES ('seed', 'x', 'x', 'x', ?)", (T,))
    con.execute("INSERT INTO collection (source_id, url, accessed_at, payload_sha256, size_bytes) "
                "VALUES (1, 'x', ?, 'x', 0)", (T,))
    con.execute("INSERT INTO parse (collection_id, parser_name, parser_version, run_at) "
                "VALUES (1, 'seed', '0', ?)", (T,))
    con.execute("INSERT INTO companies (cnpj, kind, created_at) VALUES (?, 'campaign', ?)", (CAMPAIGN_CNPJ, T))
    con.execute(
        "INSERT INTO campaign_org (company_id, person_id, cnpj, tse_candidacy_id, year, "
        "provenance_id, collected_at) VALUES (1, 1, ?, '250001', 2022, 1, ?)", (CAMPAIGN_CNPJ, T))
    yield con
    con.close()


def _supplier(con, opened, *, status="ATIVA", status_date=None, capital=None,
              nature="Sociedade Empresária Limitada", registry=True):
    cnpj = str(next(_cnpjs))
    company_id = con.execute(
        "INSERT INTO companies (cnpj, kind, created_at) VALUES (?, 'supplier', ?)", (cnpj, T)).lastrowid
    if registry:
        con.execute(
            "INSERT INTO company_registry (company_id, cnpj, legal_name, opened_at, registry_status, "
            "registry_status_date, legal_nature, share_capital_cents, provenance_id, collected_at) "
            "VALUES (?, ?, 'FORNECEDORA LTDA', ?, ?, ?, ?, ?, 1, ?)",
            (company_id, cnpj, opened, status, status_date, nature, capital, T))
    return company_id, cnpj


def _expense(con, company_id, cnpj, date, cents):
    con.execute(
        "INSERT INTO campaign_expense (campaign_org_id, cnpj, tse_candidacy_id, year, tse_expense_id, "
        "expense_date, amount_cents, supplier_cpf_cnpj, supplier_name, supplier_company_id, "
        "provenance_id, collected_at) VALUES (1, ?, '250001', 2022, ?, ?, ?, ?, 'FORNECEDORA', ?, 1, ?)",
        (CAMPAIGN_CNPJ, str(next(_expense_ids)), date, cents, cnpj, company_id, T))


def _signals(con):
    return [dict(r) for r in con.execute(
        "SELECT s.id, s.type, s.severity, s.explanation, s.amount_cents FROM signal s ORDER BY s.id")]


def test_paid_before_company_opened_is_high(world):
    cid, cnpj = _supplier(world, "2022-05-10")
    _expense(world, cid, cnpj, "2022-05-01", 300_000)
    _expense(world, cid, cnpj, "2022-05-03", 200_000)
    _expense(world, cid, cnpj, "2022-06-01", 900_000)  # after opening: not part of the signal
    world.commit()

    rep = rule.run(world)

    (sig,) = _signals(world)
    assert (sig["type"], sig["severity"]) == (rule.TYPE_BEFORE_OPENED, "high")
    assert sig["amount_cents"] == 500_000
    assert "2022-05-10" in sig["explanation"] and "2022-05-01" in sig["explanation"]
    assert "2 despesa(s)" in sig["explanation"]
    assert rep["by_type"] == {rule.TYPE_BEFORE_OPENED: 1}


def test_signal_links_company_candidate_and_evidence(world):
    cid, cnpj = _supplier(world, "2022-05-10")
    _expense(world, cid, cnpj, "2022-05-01", 300_000)
    world.commit()
    rule.run(world)

    actors = {(r["type"], r["actor_id"], r["role"]) for r in world.execute("SELECT * FROM signal_actor")}
    assert actors == {("company", cid, "supplier"), ("person", 1, "candidate")}
    evidence = world.execute("SELECT table_name, record_id FROM signal_evidence").fetchall()
    assert [(e["table_name"], e["record_id"]) for e in evidence] == [("campaign_expense", 1)]


def test_paid_after_company_became_inactive_is_high(world):
    cid, cnpj = _supplier(world, "2018-01-01", status="BAIXADA", status_date="2022-03-01")
    _expense(world, cid, cnpj, "2022-02-01", 700_000)   # before the closure: fine
    _expense(world, cid, cnpj, "2022-04-01", 400_000)
    world.commit()
    rule.run(world)

    (sig,) = _signals(world)
    assert (sig["type"], sig["severity"]) == (rule.TYPE_AFTER_INACTIVE, "high")
    assert sig["amount_cents"] == 400_000
    assert "BAIXADA" in sig["explanation"]


@pytest.mark.parametrize("status", ["INAPTA", "SUSPENSA", "NULA"])
def test_every_inactive_status_counts(world, status):
    cid, cnpj = _supplier(world, "2018-01-01", status=status, status_date="2022-03-01")
    _expense(world, cid, cnpj, "2022-04-01", 100_000)
    world.commit()
    rule.run(world)
    assert [s["type"] for s in _signals(world)] == [rule.TYPE_AFTER_INACTIVE]


def test_active_company_is_never_flagged_for_status(world):
    cid, cnpj = _supplier(world, "2018-01-01", status="ATIVA", status_date="2022-03-01")
    _expense(world, cid, cnpj, "2022-04-01", 100_000)
    world.commit()
    rule.run(world)
    assert _signals(world) == []


def test_new_company_with_low_capital_and_high_value_is_medium(world):
    cid, cnpj = _supplier(world, "2022-03-01", capital=500_000)
    _expense(world, cid, cnpj, "2022-04-01", 3_000_000)
    _expense(world, cid, cnpj, "2022-07-01", 2_500_000)
    world.commit()
    rule.run(world)

    (sig,) = _signals(world)
    assert (sig["type"], sig["severity"]) == (rule.TYPE_NEW_LOW_CAPITAL, "medium")
    assert sig["amount_cents"] == 5_500_000
    assert "31 dia(s) depois" in sig["explanation"]


@pytest.mark.parametrize("opened, capital, amount, flagged", [
    ("2022-03-01", 1_000_000, 5_000_000, True),    # capital and value exactly at the limits
    ("2022-03-01", 1_000_001, 5_000_000, False),   # capital just above
    ("2022-03-01", 1_000_000, 4_999_999, False),   # value just below
    ("2022-03-01", None, 5_000_000, False),        # capital unknown: never assumed to be low
    ("2022-04-01", 1_000_000, 5_000_000, True),    # first payment 0 days after opening
    ("2021-10-02", 1_000_000, 5_000_000, False),   # first payment 181 days after opening
    ("2021-10-03", 1_000_000, 5_000_000, True),    # first payment exactly 180 days after opening
])
def test_new_company_thresholds(world, opened, capital, amount, flagged):
    cid, cnpj = _supplier(world, opened, capital=capital)
    _expense(world, cid, cnpj, "2022-04-01", amount)
    world.commit()
    rule.run(world)
    assert [s["type"] for s in _signals(world)] == ([rule.TYPE_NEW_LOW_CAPITAL] if flagged else [])


def test_new_company_alone_or_big_value_alone_is_not_enough(world):
    new_big_capital, c1 = _supplier(world, "2022-03-01", capital=900_000_000)
    _expense(world, new_big_capital, c1, "2022-04-01", 9_000_000)
    old_low_capital, c2 = _supplier(world, "2010-03-01", capital=500_000)
    _expense(world, old_low_capital, c2, "2022-04-01", 9_000_000)
    world.commit()
    rule.run(world)
    assert _signals(world) == []


def test_campaign_cnpjs_and_party_committees_are_excluded(world):
    other_campaign, c1 = _supplier(world, "2022-05-10")
    world.execute(
        "INSERT INTO campaign_org (company_id, person_id, cnpj, tse_candidacy_id, year, "
        "provenance_id, collected_at) VALUES (?, 1, ?, '250002', 2022, 1, ?)", (other_campaign, c1, T))
    _expense(world, other_campaign, c1, "2022-05-01", 300_000)

    party, c2 = _supplier(world, "2022-05-10", nature="Órgão de Direção Regional de Partido Político")
    _expense(world, party, c2, "2022-05-01", 300_000)
    world.commit()
    rule.run(world)
    assert _signals(world) == []


def test_supplier_without_registry_data_is_not_evaluated(world):
    cid, cnpj = _supplier(world, "2022-05-10", registry=False)
    _expense(world, cid, cnpj, "2022-05-01", 300_000)
    world.commit()
    rule.run(world)
    assert _signals(world) == []


def test_evidence_is_capped_to_the_largest_expenses_but_the_text_has_the_full_count(world):
    cid, cnpj = _supplier(world, "2022-05-10")
    for i in range(rule.MAX_EVIDENCE_ROWS + 5):
        _expense(world, cid, cnpj, "2022-05-01", 1_000 + i)
    world.commit()
    rule.run(world)

    (sig,) = _signals(world)
    assert f"{rule.MAX_EVIDENCE_ROWS + 5} despesa(s)" in sig["explanation"]
    kept = [r["amount_cents"] for r in world.execute(
        "SELECT ce.amount_cents FROM signal_evidence se JOIN campaign_expense ce ON ce.id = se.record_id")]
    assert len(kept) == rule.MAX_EVIDENCE_ROWS
    assert min(kept) == 1_005  # the 5 smallest were dropped


def test_rewrite_only_rerun_does_not_duplicate(world):
    cid, cnpj = _supplier(world, "2022-05-10")
    _expense(world, cid, cnpj, "2022-05-01", 300_000)
    world.commit()

    rep1 = rule.run(world)
    rep2 = rule.run(world)

    assert rep1["signals"] == rep2["signals"] == 1
    for table in ("signal", "rule_run", "signal_evidence"):
        assert world.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1


def test_rerun_does_not_touch_other_rules_signals(world):
    world.execute("INSERT INTO rule_run (rule, rule_version, run_at, rows_generated) "
                  "VALUES ('circular_donations', '1', ?, 1)", (T,))
    world.execute("INSERT INTO signal (rule_run_id, type, severity, explanation) "
                  "VALUES (1, 'cycle', 'high', 'x')")
    world.commit()
    rule.run(world)
    assert world.execute("SELECT count(*) FROM signal WHERE type = 'cycle'").fetchone()[0] == 1


def test_never_uses_forbidden_severity_language(world):
    cid, cnpj = _supplier(world, "2022-05-10")
    _expense(world, cid, cnpj, "2022-05-01", 300_000)
    cid2, cnpj2 = _supplier(world, "2018-01-01", status="BAIXADA", status_date="2022-03-01")
    _expense(world, cid2, cnpj2, "2022-04-01", 300_000)
    cid3, cnpj3 = _supplier(world, "2022-03-01", capital=1)
    _expense(world, cid3, cnpj3, "2022-04-01", 9_000_000)
    world.commit()
    rule.run(world)

    signals = _signals(world)
    assert {s["severity"] for s in signals} <= {"low", "medium", "high"}
    text = " ".join(s["explanation"] for s in signals).upper()
    for banned in ("CONFIRMADO", "CULPADO", "CRIME", "FRAUDE"):
        assert banned not in text
