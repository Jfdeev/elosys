"""Segunda opinião de LLM (DeepSeek) sobre posts do X coletados em social_post."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..ai.deepseek import DEFAULT_MODEL, DeepSeekError, chat_json, model_chain
from ..log import RowCounter, get_logger, step
from ..util import now_utc

log = get_logger("elosys.rules.social_review")

DEFAULT_LIMIT = 100
DEFAULT_WORKERS = 8
SEVERITIES = {"low", "medium", "high"}
CATEGORIES = {
    "lgbtfobia", "racismo", "misoginia", "capacitismo", "xenofobia", "regionalismo",
    "aporofobia", "gordofobia", "antissemitismo", "intolerancia_religiosa", "etarismo_saude",
    "desumanizacao", "xingamento_pessoal",
}
GROUP_CATEGORIES = CATEGORIES - {"desumanizacao", "xingamento_pessoal"}

SYSTEM_PROMPT = (
    "Você classifica MANIFESTAÇÕES PÚBLICAS de políticos e candidatos brasileiros "
    "no X (Twitter). O texto foi pré-selecionado por um filtro de palavras que PODEM "
    "ser pejorativas — a maioria NÃO é (a palavra aparece no sentido literal, em "
    "citação, em discussão sobre o próprio preconceito, reapropriada, ou é só "
    "palavrão sem alvo). Seu trabalho é decidir, PELO CONTEXTO, se aquele texto "
    "ataca ou deprecia um GRUPO (por orientação sexual, identidade de gênero, raça/"
    "etnia, religião, deficiência, origem regional, nacionalidade, classe, corpo, "
    "idade) ou uma PESSOA com xingamento desumanizante.\n\n"
    "Julgue APENAS as palavras do autor do post. Se for resposta/citação, o texto "
    "citado é só contexto — não classifique o que o outro disse.\n\n"
    "Responda SEMPRE só com um objeto JSON:\n"
    "{\n"
    '  "ofensivo": true | false,\n'
    '  "categorias": ["lgbtfobia" | "racismo" | "misoginia" | "capacitismo" | '
    '"xenofobia" | "regionalismo" | "aporofobia" | "gordofobia" | "antissemitismo" | '
    '"intolerancia_religiosa" | "desumanizacao" | "etarismo_saude" | "xingamento_pessoal"],\n'
    '  "severity": "low" | "medium" | "high",\n'
    '  "trecho": "a parte EXATA do texto que sustenta a classificação (copie verbatim)",\n'
    '  "explicacao": "1 a 3 frases, objetivas, em português"\n'
    "}\n\n"
    "- ofensivo=false: uso literal, citação de terceiro, discussão/denúncia do "
    "preconceito, termo reapropriado pelo próprio grupo, palavrão genérico sem alvo "
    "de grupo, crítica política dura sem marcador de grupo. Deixe categorias e "
    "trecho vazios.\n"
    "- severity: low = insinuação/dogwhistle ou xingamento leve; medium = ofensa "
    "clara a um grupo; high = incitação, desumanização explícita, defesa de "
    "violência ou de discriminação.\n\n"
    "NUNCA afirme que houve crime. 'ofensivo' aqui quer dizer 'vale um humano "
    "conferir', não 'é culpado'. Na dúvida entre false e um low fraquíssimo, "
    "prefira false — o filtro anterior já é frouxo demais."
)


def run(
    con: sqlite3.Connection,
    *,
    model: str = DEFAULT_MODEL,
    limit: int = DEFAULT_LIMIT,
    refresh: bool = False,
    only_matched: bool = True,
    handles: list[str] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    with step(log, f"selecionar posts (limit {limit}, only_matched={only_matched})"):
        targets = _select_posts(con, model_chain(model), limit, refresh, only_matched, handles)
    log.info("%d posts para revisar (%d workers)", len(targets), workers)

    by_cat: dict[str, int] = {}
    reviewed = flagged = errors = 0
    rc = RowCounter(log, "posts revisados", every=25)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_review_one, row["id"], _user_prompt(row), model): row["id"]
            for row in targets
        }
        for fut in as_completed(futs):
            post_id = futs[fut]
            rc.tick()
            try:
                user_prompt, out = fut.result()
            except DeepSeekError as e:
                errors += 1
                log.warning("post %d: %s", post_id, e)
                if errors >= 10 and errors > reviewed:
                    log.error("erros demais (%d) e nenhum sucesso — abortando", errors)
                    con.commit()
                    return _summary(reviewed, flagged, by_cat, errors, aborted=True)
                continue

            offensive, categories, severity, quote, explanation = _parse(out["data"])
            con.execute(
                "INSERT INTO social_post_review (social_post_id, model, reviewed_at, categories, "
                "severity, is_offensive, quote, explanation, prompt, raw_response, tokens_prompt, "
                "tokens_completion) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (social_post_id, model) DO UPDATE SET "
                "reviewed_at=excluded.reviewed_at, categories=excluded.categories, "
                "severity=excluded.severity, is_offensive=excluded.is_offensive, quote=excluded.quote, "
                "explanation=excluded.explanation, prompt=excluded.prompt, "
                "raw_response=excluded.raw_response, tokens_prompt=excluded.tokens_prompt, "
                "tokens_completion=excluded.tokens_completion",
                (
                    post_id, out.get("model", model), now_utc(), json.dumps(categories, ensure_ascii=False),
                    severity, 1 if offensive else 0, quote, explanation, user_prompt, out["raw"],
                    out["prompt_tokens"], out["completion_tokens"],
                ),
            )
            reviewed += 1
            if offensive:
                flagged += 1
                for c in categories:
                    by_cat[c] = by_cat.get(c, 0) + 1
            if reviewed % 25 == 0:
                con.commit()
    rc.done()
    con.commit()
    return _summary(reviewed, flagged, by_cat, errors)


def _review_one(post_id: int, user_prompt: str, model: str) -> tuple[str, dict]:
    return user_prompt, chat_json(SYSTEM_PROMPT, user_prompt, model=model)


def _summary(reviewed: int, flagged: int, by_cat: dict, errors: int, *, aborted: bool = False) -> dict:
    out = {"reviewed": reviewed, "flagged": flagged, "by_category": by_cat, "errors": errors}
    if aborted:
        out["aborted"] = True
    log.info("done: %d revisados, %d sinalizados %s (%d erros)%s",
             reviewed, flagged, by_cat, errors, " — ABORTADO" if aborted else "")
    return out


def _select_posts(
    con: sqlite3.Connection, models: list[str], limit: int, refresh: bool, only_matched: bool,
    handles: list[str] | None,
) -> list[sqlite3.Row]:
    marks = ", ".join("?" * len(models))
    if refresh:
        con.execute(f"DELETE FROM social_post_review WHERE model IN ({marks})", models)  # noqa: S608
    where = [f"p.id NOT IN (SELECT social_post_id FROM social_post_review WHERE model IN ({marks}))"]
    params: list = [*models]
    if only_matched:
        where.append("p.matched_terms != '[]'")
    if handles:
        marks = ", ".join("?" * len(handles))
        where.append(f"a.handle IN ({marks})")
        params += [h.lstrip("@").lower() for h in handles]
    params.append(limit)
    return con.execute(
        f"SELECT p.id, p.kind, p.text, p.reply_to_handle, p.matched_terms, p.posted_at, "  # noqa: S608
        f"       a.handle "
        f"FROM social_post p JOIN social_account a ON a.id = p.social_account_id "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY p.id LIMIT ?",
        params,
    ).fetchall()


def _user_prompt(row: sqlite3.Row) -> str:
    ctx = f"Conta: @{row['handle']}. Tipo: {row['kind']}."
    if row["kind"] == "reply" and row["reply_to_handle"]:
        ctx += f" (resposta a @{row['reply_to_handle']})"
    try:
        terms = ", ".join(json.loads(row["matched_terms"]))
    except (TypeError, ValueError):
        terms = ""
    return (
        f"{ctx}\n"
        f"Termos do filtro que apareceram: {terms}\n\n"
        f"Texto do post (verbatim):\n\"\"\"\n{row['text']}\n\"\"\"\n\n"
        "Classifique no formato JSON pedido, julgando só as palavras do autor."
    )


def _parse(data: dict) -> tuple[bool, list[str], str | None, str | None, str]:
    offensive = bool(data.get("ofensivo") if "ofensivo" in data else data.get("offensive"))
    cats = data.get("categorias") or data.get("categories") or []
    if not isinstance(cats, list):
        cats = [str(cats)]
    cats = [c for c in (str(x).strip().lower() for x in cats) if c in CATEGORIES]
    severity = str(data.get("severity") or "").strip().lower()
    if severity not in SEVERITIES:
        severity = None
    quote = (data.get("trecho") or data.get("quote") or "").strip() or None
    explanation = str(data.get("explicacao") or data.get("explanation") or "").strip()
    if not offensive:
        cats, quote = [], None
    if offensive and not severity:
        severity = "low"
    return offensive, cats, severity, quote, explanation or "(sem explicação)"
