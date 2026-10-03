"""Detection rule: disproportionate_expense, against synthetic campaign_expense rows."""

from __future__ import annotations

from elosys.db import connect, create_schema
from elosys.rules import disproportionate_expense as rule

T = "2026-01-01T00:00:00Z"


def _seed_common(con):
    con.execute("INSERT INTO people (cpf, cpf_trusted, canonical_name, created_at) "
                "VALUES ('11144477735', 1, 'JOAO DA SILVA', ?)", (T,))
    pid = con.execute("SELECT id FROM people").fetchone()["id"]
    con.execute("INSERT INTO companies (cnpj, kind, created_at) VALUES ('98765432000155', 'supplier', ?)", (T,))
    company_id = con.execute("SELECT id FROM companies").fetchone()["id"]
    con.execute("INSERT INTO source (name, agency, type, base_url, created_at) "
                "VALUES ('seed', 'x', 'x', 'x', ?)", (T,))
    con.execute("INSERT INTO collection (source_id, url, accessed_at, payload_sha256, size_bytes) "
                "VALUES (1, 'x', ?, 'x', 0)", (T,))
    con.execute("INSERT INTO parse (collection_id, parser_name, parser_version, run_at) "
                "VALUES (1, 'seed', '0', ?)", (T,))
    con.execute(
        "INSERT INTO campaign_org (company_id, person_id, cnpj, tse_candidacy_id, year, "
        "provenance_id, collected_at) VALUES (?, ?, '40430149000110', '250001', 2022, 1, ?)",
        (company_id, pid, T))
    org_id = con.execute("SELECT id FROM campaign_org").fetchone()["id"]
    return pid, company_id, org_id


def _insert_expense(con, org_id, company_id, description, amount_cents, expense_id, year=2022):
    con.execute(
        "INSERT INTO campaign_expense (campaign_org_id, cnpj, tse_candidacy_id, year, "
        "tse_expense_id, description, amount_cents, supplier_company_id, "
        "provenance_id, collected_at) "
        "VALUES (?, '40430149000110', '250001', ?, ?, ?, ?, ?, 1, ?)",
        (org_id, year, expense_id, description, amount_cents, company_id, T))


def _add_org(con, company_id, year, office, candidacy_id):
    cur = con.execute(
        "INSERT INTO campaign_org (company_id, cnpj, tse_candidacy_id, year, office, "
        "provenance_id, collected_at) VALUES (?, '40430149000110', ?, ?, ?, 1, ?)",
        (company_id, candidacy_id, year, office, T))
    return cur.lastrowid


def _flagged(con):
    """{tse_expense_id: (severity, explanation)} for every signal."""
    return {
        r["tse_expense_id"]: (r["severity"], r["explanation"])
        for r in con.execute(
            "SELECT ce.tse_expense_id, s.severity, s.explanation FROM signal s "
            "JOIN signal_evidence se ON se.signal_id = s.id "
            "JOIN campaign_expense ce ON ce.id = se.record_id")
    }


def test_flags_cheap_item_over_threshold(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    pid, company_id, org_id = _seed_common(con)

    _insert_expense(con, org_id, company_id, "CANETAS E BLOCOS DE ANOTACAO", 1_500_000, "1")
    _insert_expense(con, org_id, company_id, "ADESIVOS, PRAGUINHA, PRAGAO, PERFURADOS", 153_900_000, "2")
    _insert_expense(con, org_id, company_id, "CANETA AZUL", 20_000, "3")
    _insert_expense(con, org_id, company_id, "LOCACAO DE VEICULO PARA CAMPANHA", 8_000_000, "4")
    con.commit()

    rep = rule.run(con)
    assert rep["signals"] == 2
    assert rep["by_severity"] == {"medium": 1, "high": 1}

    signals = con.execute(
        "SELECT s.type, s.severity, s.explanation, se.record_id "
        "FROM signal s JOIN signal_evidence se ON se.signal_id = s.id "
        "ORDER BY se.record_id"
    ).fetchall()
    assert [dict(r)["severity"] for r in signals] == ["medium", "high"]
    assert all(r["type"] == "cheap_item_high_value" for r in signals)
    assert "R$" in signals[0]["explanation"]

    actor_types = {
        (r["type"], r["role"]) for r in con.execute(
            "SELECT sa.type, sa.role FROM signal_actor sa "
            "JOIN signal s ON s.id = sa.signal_id "
            "JOIN signal_evidence se ON se.signal_id = s.id WHERE se.record_id = 1"
        )
    }
    assert ("person", "candidate") in actor_types
    assert ("company", "supplier") in actor_types
    con.close()


def test_evidence_traces_to_provenance(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, org_id = _seed_common(con)
    _insert_expense(con, org_id, company_id, "CRACHAS PERSONALIZADOS", 900_000, "1")
    con.commit()

    rule.run(con)
    src = con.execute(
        "SELECT src.name FROM signal_evidence se "
        "JOIN campaign_expense ce ON ce.id = se.record_id "
        "JOIN parse p ON p.id = ce.provenance_id "
        "JOIN collection c ON c.id = p.collection_id "
        "JOIN source src ON src.id = c.source_id "
        "WHERE se.table_name = 'campaign_expense'"
    ).fetchone()
    assert src["name"] == "seed"
    con.close()


def test_rewrite_only_rerun_does_not_duplicate(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, org_id = _seed_common(con)
    _insert_expense(con, org_id, company_id, "PASTAS E ENVELOPES", 900_000, "1")
    con.commit()

    rep1 = rule.run(con)
    rep2 = rule.run(con)
    assert rep1["signals"] == rep2["signals"] == 1
    assert con.execute("SELECT count(*) FROM signal").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM rule_run").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM signal_evidence").fetchone()[0] == 1
    con.close()


def test_median_based_threshold_for_high_sample_category(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, org_id = _seed_common(con)

    for i in range(20):
        amount = 10_000 if i != 0 else 20_000
        _insert_expense(con, org_id, company_id, "CANETAS PERSONALIZADAS", amount, f"base-{i}")
    _insert_expense(con, org_id, company_id, "CANETAS PERSONALIZADAS", 200_000, "medium-case")
    _insert_expense(con, org_id, company_id, "CANETAS PERSONALIZADAS", 500_000, "high-case")
    _insert_expense(con, org_id, company_id, "CANETAS PERSONALIZADAS", 30_000, "not-flagged")
    con.commit()

    rule.run(con)
    flagged = {
        r["record_id"]: r["severity"]
        for r in con.execute(
            "SELECT se.record_id, s.severity FROM signal s "
            "JOIN signal_evidence se ON se.signal_id = s.id"
        )
    }
    ids_by_expense = dict(con.execute("SELECT tse_expense_id, id FROM campaign_expense"))
    assert flagged.get(ids_by_expense["medium-case"]) == "medium"
    assert flagged.get(ids_by_expense["high-case"]) == "high"
    assert ids_by_expense["not-flagged"] not in flagged
    assert all(ids_by_expense[f"base-{i}"] not in flagged for i in range(20))
    con.close()


def test_never_uses_forbidden_severity_language(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, org_id = _seed_common(con)
    _insert_expense(con, org_id, company_id, "REGUAS E MARCADOR DE TEXTO", 900_000, "1")
    con.commit()
    rule.run(con)

    severities = {r["severity"] for r in con.execute("SELECT DISTINCT severity FROM signal")}
    assert severities <= {"low", "medium", "high"}
    explanations = " ".join(r["explanation"] for r in con.execute("SELECT explanation FROM signal"))
    for banned in ("CONFIRMADO", "CULPADO", "CRIME", "FRAUDE"):
        assert banned not in explanations.upper()
    con.close()


def test_median_is_computed_per_year_and_office(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, _ = _seed_common(con)
    vereador = _add_org(con, company_id, 2020, "Vereador", "v1")
    deputado = _add_org(con, company_id, 2022, "Deputado Federal", "d1")

    for i in range(100):
        _insert_expense(con, vereador, company_id, "ADESIVO", 10_000, f"v-{i}", year=2020)
    for i in range(20):
        _insert_expense(con, deputado, company_id, "ADESIVO", 200_000, f"d-{i}", year=2022)
    _insert_expense(con, vereador, company_id, "ADESIVO", 500_000, "v-outlier", year=2020)
    con.commit()

    rule.run(con)
    flagged = _flagged(con)

    # Under one pooled median (10k) the typical federal-deputy sticker (200k) would be flagged.
    assert not any(k.startswith("d-") for k in flagged)
    severity, explanation = flagged["v-outlier"]
    assert severity == "high"
    assert "Vereador 2020" in explanation
    con.close()


def test_small_office_group_falls_back_to_year_median(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, _ = _seed_common(con)
    prefeito = _add_org(con, company_id, 2022, "Prefeito", "p1")
    senador = _add_org(con, company_id, 2022, "Senador", "s1")

    for i in range(20):
        _insert_expense(con, prefeito, company_id, "ADESIVO", 10_000, f"p-{i}")
    _insert_expense(con, senador, company_id, "ADESIVO", 200_000, "s-outlier")
    con.commit()

    rule.run(con)
    severity, explanation = _flagged(con)["s-outlier"]
    assert severity == "medium"
    assert "eleição de 2022" in explanation
    con.close()


def test_multi_item_descriptions_do_not_pollute_the_median(tmp_path):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _, company_id, org_id = _seed_common(con)

    for i in range(20):
        _insert_expense(con, org_id, company_id, "CANETA", 10_000, f"c-{i}")
    for i in range(30):
        _insert_expense(con, org_id, company_id, "CANETA E ADESIVO", 1_000_000, f"bundle-{i}")
    _insert_expense(con, org_id, company_id, "CANETA", 200_000, "c-outlier")
    con.commit()

    rule.run(con)
    flagged = _flagged(con)

    assert "20x a mediana" in flagged["c-outlier"][1]
    assert all(flagged[f"bundle-{i}"][0] == "medium" for i in range(30))
    assert "vários itens" in flagged["bundle-0"][1]
    con.close()


def test_categories_in_counts_an_item_once():
    assert rule._categories_in("LAPISEIRA 0.7") == {"LAPISEIRA"}
    assert rule._categories_in("LAPIS E LAPISEIRA") == {"LAPIS", "LAPISEIRA"}
    assert rule._categories_in("CANETA E ADESIVO") == {"CANETA", "ADESIVO"}
    assert rule._categories_in("LOCACAO DE VEICULO") == set()
