"""Detection rule: doação circular — money that leaves a politician's own"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict

from ..db import connect, create_schema
from ..log import RowCounter, get_logger, step
from ..util import git_commit, now_utc
from .common import reset_rule_signals

log = get_logger("elosys.rules.circular_donations")

RULE_NAME = "circular_donations"
RULE_VERSION = "1.1"

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_FANOUT = 400
DEFAULT_MIN_AMOUNT_CENTS = 1_000_000

HIGH_SEVERITY_MAX_LEN = 3

EVIDENCE_ROWS_PER_EDGE = 10


def _build_graph(
    con: sqlite3.Connection,
) -> tuple[list[str], dict[int, list[int]], dict[tuple[int, int], str], dict[tuple[int, int], int]]:
    node_id: dict[str, int] = {}
    id_to_node: list[str] = []

    def get_id(key: str) -> int:
        i = node_id.get(key)
        if i is None:
            i = len(id_to_node)
            node_id[key] = i
            id_to_node.append(key)
        return i

    with step(log, "resolve campaign_org -> candidate cpf"):
        org_cpf: dict[int, str] = {
            row[0]: row[1]
            for row in con.execute(
                "SELECT co.id, p.cpf FROM campaign_org co "
                "JOIN people p ON p.id = co.person_id WHERE p.cpf IS NOT NULL"
            )
        }
    log.info("  %s candidacies resolved to a cpf", f"{len(org_cpf):,}")

    edges: set[tuple[int, int]] = set()
    edge_kind: dict[tuple[int, int], str] = {}
    edge_amount: dict[tuple[int, int], int] = {}

    def add_edge(src: str, dst: str, kind: str, amount: int | None) -> None:
        if src == dst:
            return
        s, d = get_id(src), get_id(dst)
        key = (s, d)
        amount = amount or 0
        if key in edges:
            prev = edge_kind[key]
            if prev != kind:
                edge_kind[key] = "both"
            edge_amount[key] += amount
        else:
            edges.add(key)
            edge_kind[key] = kind
            edge_amount[key] = amount

    with step(log, "stream campaign_donation (donor -> candidate)"):
        rc = RowCounter(log, "donation rows scanned", every=1_000_000)
        cur = con.execute(
            "SELECT campaign_org_id, donor_cpf_cnpj, amount_cents FROM campaign_donation "
            "WHERE donor_cpf_cnpj IS NOT NULL AND campaign_org_id IS NOT NULL"
        )
        for co_id, donor, amount in cur:
            rc.tick()
            cpf = org_cpf.get(co_id)
            if cpf is not None:
                add_edge(donor, cpf, "donation", amount)
        rc.done()

    with step(log, "stream campaign_expense (candidate -> supplier)"):
        rc = RowCounter(log, "expense rows scanned", every=1_000_000)
        cur = con.execute(
            "SELECT campaign_org_id, supplier_cpf_cnpj, amount_cents FROM campaign_expense "
            "WHERE supplier_cpf_cnpj IS NOT NULL AND campaign_org_id IS NOT NULL"
        )
        for co_id, supplier, amount in cur:
            rc.tick()
            cpf = org_cpf.get(co_id)
            if cpf is not None:
                add_edge(cpf, supplier, "payment", amount)
        rc.done()

    log.info("  graph: %s nodes, %s edges", f"{len(id_to_node):,}", f"{len(edges):,}")

    adj: dict[int, list[int]] = defaultdict(list)
    for s, d in edges:
        adj[s].append(d)

    return id_to_node, adj, edge_kind, edge_amount


def _tarjan_scc(n: int, adj: dict[int, list[int]]) -> list[list[int]]:
    index_of = [-1] * n
    lowlink = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    sccs: list[list[int]] = []
    counter = 0

    for start in range(n):
        if index_of[start] != -1:
            continue
        work: list[tuple[int, int]] = [(start, 0)]
        index_of[start] = counter
        lowlink[start] = counter
        counter += 1
        stack.append(start)
        on_stack[start] = True

        while work:
            v, i = work[-1]
            neighbors = adj.get(v, ())
            if i < len(neighbors):
                w = neighbors[i]
                work[-1] = (v, i + 1)
                if index_of[w] == -1:
                    index_of[w] = counter
                    lowlink[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index_of[w])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == index_of[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)
    return sccs


def _find_cycles_in_scc(
    scc_nodes: list[int],
    adj: dict[int, list[int]],
    max_depth: int,
    max_fanout: int,
) -> tuple[list[list[int]], int]:
    members = set(scc_nodes)
    induced: dict[int, list[int]] = {
        v: [w for w in adj.get(v, ()) if w in members] for v in scc_nodes
    }
    state = _CycleSearchState(induced, max_depth, max_fanout, cycles=[], hub_skipped=0)

    for start in scc_nodes:
        _dfs_from(state, start, start, [start], {start})

    return state.cycles, state.hub_skipped


class _CycleSearchState:
    def __init__(self, induced: dict[int, list[int]], max_depth: int, max_fanout: int,
                 cycles: list[list[int]], hub_skipped: int) -> None:
        self.induced = induced
        self.max_depth = max_depth
        self.max_fanout = max_fanout
        self.cycles = cycles
        self.hub_skipped = hub_skipped


def _dfs_from(state: _CycleSearchState, start: int, current: int, path: list[int], visited: set[int]) -> None:
    neighbors = state.induced.get(current, ())
    if len(neighbors) > state.max_fanout and current != start:
        state.hub_skipped += 1
        return
    for nxt in neighbors:
        if nxt == start and len(path) >= 2:
            state.cycles.append(list(path))
        elif len(path) < state.max_depth and nxt > start and nxt not in visited:
            visited.add(nxt)
            path.append(nxt)
            _dfs_from(state, start, nxt, path, visited)
            path.pop()
            visited.discard(nxt)


def _severity(cycle_len: int) -> str:
    return "high" if cycle_len <= HIGH_SEVERITY_MAX_LEN else "medium"


def run(con: sqlite3.Connection, *, max_depth: int = DEFAULT_MAX_DEPTH,
        max_fanout: int = DEFAULT_MAX_FANOUT,
        min_amount_cents: int = DEFAULT_MIN_AMOUNT_CENTS) -> dict:
    with step(log, "reset circular_donations signals"):
        reset_rule_signals(con, RULE_NAME)

    id_to_node, adj, edge_kind, edge_amount = _build_graph(con)
    n = len(id_to_node)

    with step(log, "Tarjan SCC"):
        sccs = _tarjan_scc(n, adj)
    nontrivial = [s for s in sccs if len(s) > 1]
    log.info("  %s SCCs total, %s with more than one node (candidates for a cycle)",
              f"{len(sccs):,}", f"{len(nontrivial):,}")

    all_cycles: list[list[int]] = []
    hub_skipped_total = 0
    with step(log, f"enumerate simple cycles (max_depth={max_depth}, max_fanout={max_fanout})"):
        for scc in nontrivial:
            cycles, hub_skipped = _find_cycles_in_scc(scc, adj, max_depth, max_fanout)
            all_cycles.extend(cycles)
            hub_skipped_total += hub_skipped
    log.info("  %s cycles found (%s hub-node branches skipped)",
              f"{len(all_cycles):,}", f"{hub_skipped_total:,}")

    params = {"max_depth": max_depth, "max_fanout": max_fanout, "min_amount_cents": min_amount_cents}
    cur = con.execute(
        "INSERT INTO rule_run (rule, rule_version, code_commit, params, run_at, rows_generated) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (RULE_NAME, RULE_VERSION, git_commit(), json.dumps(params, ensure_ascii=False), now_utc()),
    )
    rule_run_id = int(cur.lastrowid)

    generated = 0
    below_floor = 0
    cache = _WriteCache()
    with step(log, "write signals"):
        rc = RowCounter(log, "signals written", every=20_000)
        for cycle in all_cycles:
            cpfs = [id_to_node[i] for i in cycle]
            wrote = _write_cycle_signal(
                con, rule_run_id, cycle, cpfs, edge_kind, edge_amount, cache, min_amount_cents
            )
            generated += wrote
            below_floor += 0 if wrote else 1
            rc.tick()
        rc.done()
    log.info("  %s cycles skipped (total movement below R$ %.2f)",
             f"{below_floor:,}", min_amount_cents / 100)
    con.execute("UPDATE rule_run SET rows_generated = ? WHERE id = ?", (generated, rule_run_id))
    con.commit()

    by_severity = dict(con.execute(
        "SELECT severity, count(*) FROM signal WHERE rule_run_id = ? GROUP BY severity",
        (rule_run_id,),
    ).fetchall())
    log.info("done: %s sinais gerados (rule_run %d) — %s",
             f"{generated:,}", rule_run_id, by_severity)
    return {
        "rule_run_id": rule_run_id,
        "nodes": n,
        "edges": len(edge_kind),
        "sccs_nontrivial": len(nontrivial),
        "cycles_found": len(all_cycles),
        "hub_branches_skipped": hub_skipped_total,
        "signals": generated,
        "by_severity": by_severity,
        "params": params,
    }


class _WriteCache:
    def __init__(self) -> None:
        self.name: dict[str, str | None] = {}
        self.actor: dict[str, tuple[str, int] | None] = {}
        self.evidence: dict[tuple[str, str, str], list[tuple[str, int]]] = {}


def _name_for(con: sqlite3.Connection, cpf_cnpj: str, cache: _WriteCache) -> str | None:
    if cpf_cnpj in cache.name:
        return cache.name[cpf_cnpj]
    if len(cpf_cnpj) == 11:
        row = con.execute("SELECT canonical_name FROM people WHERE cpf = ?", (cpf_cnpj,)).fetchone()
    else:
        row = con.execute(
            "SELECT coalesce(cr.legal_name, c.legal_name) FROM companies c "
            "LEFT JOIN company_registry cr ON cr.company_id = c.id WHERE c.cnpj = ?",
            (cpf_cnpj,),
        ).fetchone()
    name = row[0] if row else None
    cache.name[cpf_cnpj] = name
    return name


def _actor_for(con: sqlite3.Connection, cpf_cnpj: str, cache: _WriteCache) -> tuple[str, int] | None:
    if cpf_cnpj in cache.actor:
        return cache.actor[cpf_cnpj]
    kind = "person" if len(cpf_cnpj) == 11 else "company"
    table, col = ("people", "cpf") if kind == "person" else ("companies", "cnpj")
    row = con.execute(f"SELECT id FROM {table} WHERE {col} = ?", (cpf_cnpj,)).fetchone()  # noqa: S608
    actor = (kind, row[0]) if row else None
    cache.actor[cpf_cnpj] = actor
    return actor


def _evidence_for(
    con: sqlite3.Connection, src_cpf: str, dst_cpf: str, kind: str, cache: _WriteCache
) -> list[tuple[str, int]]:
    cache_key = (src_cpf, dst_cpf, kind)
    if cache_key in cache.evidence:
        return cache.evidence[cache_key]
    rows: list[tuple[str, int]] = []
    if kind in ("donation", "both"):
        rows.extend(
            ("campaign_donation", r[0])
            for r in con.execute(
                "SELECT d.id FROM campaign_donation d JOIN campaign_org co ON co.id = d.campaign_org_id "
                "JOIN people p ON p.id = co.person_id "
                "WHERE d.donor_cpf_cnpj = ? AND p.cpf = ? LIMIT ?",
                (src_cpf, dst_cpf, EVIDENCE_ROWS_PER_EDGE),
            )
        )
    if kind in ("payment", "both"):
        rows.extend(
            ("campaign_expense", r[0])
            for r in con.execute(
                "SELECT e.id FROM campaign_expense e JOIN campaign_org co ON co.id = e.campaign_org_id "
                "JOIN people p ON p.id = co.person_id "
                "WHERE p.cpf = ? AND e.supplier_cpf_cnpj = ? LIMIT ?",
                (src_cpf, dst_cpf, EVIDENCE_ROWS_PER_EDGE),
            )
        )
    cache.evidence[cache_key] = rows
    return rows


def _write_cycle_signal(
    con: sqlite3.Connection,
    rule_run_id: int,
    cycle_ids: list[int],
    cpfs: list[str],
    edge_kind: dict[tuple[int, int], str],
    edge_amount: dict[tuple[int, int], int],
    cache: _WriteCache,
    min_amount_cents: int,
) -> int:
    n = len(cpfs)
    total_amount = sum(
        edge_amount.get((cycle_ids[i], cycle_ids[(i + 1) % n]), 0) for i in range(n)
    )
    if total_amount < min_amount_cents:
        return 0
    chain = " -> ".join(f"{cpfs[i]} ({_name_for(con, cpfs[i], cache) or 'sem nome'})" for i in range(n))
    explanation = (
        f"Loop de movimentação de campanha entre {n} entidades, R$ {total_amount / 100:,.2f} "
        f"movimentados no total: {chain} -> {cpfs[0]}. Cada seta é uma doação recebida ou uma "
        "despesa paga por uma campanha à seguinte da cadeia, fechando um ciclo. Pode ser "
        "coincidência entre campanhas de uma mesma coligação, um ressarcimento, ou merecer uma "
        "checagem manual mais de perto — não é, por si só, indício de irregularidade."
    )
    sig_cur = con.execute(
        "INSERT INTO signal (rule_run_id, type, severity, explanation, amount_cents, path_length) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (rule_run_id, "circular_donation", _severity(n), explanation, total_amount, n),
    )
    signal_id = int(sig_cur.lastrowid)

    actors = [(*a, "cycle_member") for cpf in cpfs if (a := _actor_for(con, cpf, cache)) is not None]
    con.executemany(
        "INSERT OR IGNORE INTO signal_actor (signal_id, type, actor_id, role) VALUES (?, ?, ?, ?)",
        [(signal_id, t, aid, role) for t, aid, role in actors],
    )

    evidence: list[tuple[int, str, int]] = []
    for i in range(n):
        s, d = cycle_ids[i], cycle_ids[(i + 1) % n]
        kind = edge_kind.get((s, d))
        if kind is None:
            continue
        src_cpf, dst_cpf = cpfs[i], cpfs[(i + 1) % n]
        evidence.extend(
            (signal_id, table, record_id) for table, record_id in _evidence_for(con, src_cpf, dst_cpf, kind, cache)
        )
    con.executemany(
        "INSERT OR IGNORE INTO signal_evidence (signal_id, table_name, record_id) VALUES (?, ?, ?)",
        evidence,
    )
    return 1


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m elosys.rules.circular_donations")
    p.add_argument("--db", default="elosys.db")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help=f"max cycle length in hops (default: {DEFAULT_MAX_DEPTH})")
    p.add_argument("--max-fanout", type=int, default=DEFAULT_MAX_FANOUT,
                   help=f"skip branching through nodes with more edges than this (default: {DEFAULT_MAX_FANOUT})")
    p.add_argument("--min-amount-brl", type=float, default=DEFAULT_MIN_AMOUNT_CENTS / 100,
                   help="skip a cycle whose total movement is below this many reais "
                        f"(default: {DEFAULT_MIN_AMOUNT_CENTS / 100:.2f})")
    args = p.parse_args(argv)

    create_schema(args.db)
    con = connect(args.db, write=True)
    try:
        report = run(con, max_depth=args.max_depth, max_fanout=args.max_fanout,
                    min_amount_cents=round(args.min_amount_brl * 100))
    finally:
        con.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
