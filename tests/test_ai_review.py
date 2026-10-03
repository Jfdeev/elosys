"""ai_review: the LLM call is mocked; we check signal selection, prompt"""

from __future__ import annotations

from elosys.db import connect, create_schema
from elosys.rules import ai_review, circular_donations

T = "2026-01-01T00:00:00Z"


def _seed_two_node_cycle(con):
    con.execute("INSERT INTO source (name, agency, type, base_url, created_at) VALUES ('seed','x','x','x',?)", (T,))
    con.execute("INSERT INTO collection (source_id, url, accessed_at, payload_sha256, size_bytes) "
                "VALUES (1,'x',?,'x',0)", (T,))
    con.execute("INSERT INTO parse (collection_id, parser_name, parser_version, run_at) VALUES (1,'seed','0',?)", (T,))
    con.execute("INSERT INTO people (cpf, cpf_trusted, canonical_name, created_at) "
                "VALUES ('22255588846', 1, 'CANDIDATA B', ?)", (T,))
    pid = con.execute("SELECT id FROM people").fetchone()["id"]
    con.execute("INSERT INTO companies (cnpj, kind, created_at) VALUES ('30000100000000','campaign',?)", (T,))
    cid = con.execute("SELECT id FROM companies").fetchone()["id"]
    con.execute("INSERT INTO campaign_org (company_id, person_id, cnpj, tse_candidacy_id, year, "
                "provenance_id, collected_at) VALUES (?,?,'30000100000000','300001',2022,1,?)", (cid, pid, T))
    org = con.execute("SELECT id FROM campaign_org").fetchone()["id"]
    con.execute("INSERT INTO campaign_donation (campaign_org_id, cnpj, year, tse_receipt_id, amount_cents, "
                "donor_cpf_cnpj, donor_name, provenance_id, collected_at) "
                "VALUES (?,'x',2022,'r1',1000000,'11144477735','DOADOR A',1,?)", (org, T))
    con.execute("INSERT INTO campaign_expense (campaign_org_id, cnpj, year, tse_expense_id, amount_cents, "
                "supplier_cpf_cnpj, supplier_name, provenance_id, collected_at) "
                "VALUES (?,'x',2022,'e1',800000,'11144477735','DOADOR A',1,?)", (org, T))
    con.execute("INSERT INTO politician_history (person_id, cpf_trusted, year, office, party_abbr, state, "
                "provenance_id, collected_at) VALUES (?,1,2022,'DEPUTADO FEDERAL','PT','SP',1,?)", (pid, T))
    con.commit()
    circular_donations.run(con)


def _fake_chat(verdict="bizarro"):
    def _inner(system_prompt, user_prompt, *, model="deepseek-chat", temperature=0.2, timeout=120):
        assert "DOAÇÃO CIRCULAR" in user_prompt
        assert "CANDIDATA B" in user_prompt
        return {
            "data": {
                "verdict": verdict,
                "confianca": "media",
                "explicacao": "Loop fechado entre um doador e a campanha, sem relação partidária óbvia.",
                "fatos": ["doador e fornecedor são a mesma pessoa", "ida e volta no mesmo ano"],
            },
            "raw": f'{{"verdict": "{verdict}"}}',
            "prompt_tokens": 300,
            "completion_tokens": 80,
        }
    return _inner


def test_writes_review_row(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _seed_two_node_cycle(con)
    monkeypatch.setattr(ai_review, "chat_json", _fake_chat())
    monkeypatch.setattr(ai_review, "DELAY_SECONDS", 0)

    rep = ai_review.run(con, limit=10, rules=("circular_donations",))
    assert rep["reviewed"] == 1
    assert rep["by_verdict"] == {"bizarro": 1}

    row = con.execute("SELECT * FROM signal_ai_review").fetchone()
    assert row["model"] == "deepseek-chat"
    assert row["verdict"] == "bizarro"
    assert row["confidence"] == "media"
    assert "Loop fechado" in row["explanation"]
    assert "mesma pessoa" in row["facts"]
    assert "DOAÇÃO CIRCULAR" in row["prompt"]
    assert row["tokens_prompt"] == 300
    con.close()


def test_incremental_and_refresh(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _seed_two_node_cycle(con)
    monkeypatch.setattr(ai_review, "DELAY_SECONDS", 0)

    monkeypatch.setattr(ai_review, "chat_json", _fake_chat("bizarro"))
    ai_review.run(con, limit=10, rules=("circular_donations",))
    rep2 = ai_review.run(con, limit=10, rules=("circular_donations",))
    assert rep2["reviewed"] == 0
    assert con.execute("SELECT count(*) FROM signal_ai_review").fetchone()[0] == 1

    monkeypatch.setattr(ai_review, "chat_json", _fake_chat("plausivel"))
    rep3 = ai_review.run(con, limit=10, rules=("circular_donations",), refresh=True)
    assert rep3["reviewed"] == 1
    assert con.execute("SELECT count(*) FROM signal_ai_review").fetchone()[0] == 1
    assert con.execute("SELECT verdict FROM signal_ai_review").fetchone()[0] == "plausivel"
    con.close()


def test_bad_verdict_falls_back_to_inconclusivo():
    v, c, expl, facts = ai_review._parse({"verdict": "CULPADO", "explicacao": "x"})
    assert v == "inconclusivo"
    assert facts == []
    assert c is None


def test_api_error_is_survived_not_crashed(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _seed_two_node_cycle(con)
    monkeypatch.setattr(ai_review, "DELAY_SECONDS", 0)

    def _boom(*a, **k):
        raise ai_review.DeepSeekError("HTTP 401: invalid key")

    monkeypatch.setattr(ai_review, "chat_json", _boom)
    rep = ai_review.run(con, limit=10, rules=("circular_donations",))
    assert rep["reviewed"] == 0
    assert rep["errors"] == 1
    assert con.execute("SELECT count(*) FROM signal_ai_review").fetchone()[0] == 0
    con.close()


def test_records_the_model_that_actually_answered(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _seed_two_node_cycle(con)
    monkeypatch.setattr(ai_review, "DELAY_SECONDS", 0)

    inner = _fake_chat()
    monkeypatch.setattr(
        ai_review, "chat_json",
        lambda *a, **k: {**inner(*a, **k), "model": "fallback-model"},
    )
    ai_review.run(con, limit=10, rules=("circular_donations",))

    assert con.execute("SELECT model FROM signal_ai_review").fetchone()[0] == "fallback-model"
    con.close()


def test_signal_reviewed_by_a_fallback_model_is_not_reviewed_again(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    path = tmp_path / "t.db"
    create_schema(path)
    con = connect(path, write=True)
    _seed_two_node_cycle(con)
    monkeypatch.setattr(ai_review, "DELAY_SECONDS", 0)

    calls = []
    inner = _fake_chat()

    def _chat(*a, **k):
        calls.append(k["model"])
        return {**inner(*a, **k), "model": "gemini-3.5-flash-lite"}  # first model was out of quota

    monkeypatch.setattr(ai_review, "chat_json", _chat)

    ai_review.run(con, model="gemini-3.1-flash-lite", limit=10, rules=("circular_donations",))
    rep2 = ai_review.run(con, model="gemini-3.1-flash-lite", limit=10, rules=("circular_donations",))

    assert rep2["reviewed"] == 0
    assert calls == ["gemini-3.1-flash-lite"]
    con.close()
