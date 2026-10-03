"""Helpers shared by the detection rules."""

from __future__ import annotations

import sqlite3


def reset_rule_signals(con: sqlite3.Connection, rule: str) -> None:
    """Delete everything a previous run of `rule` produced (rewrite-only: each run starts clean)."""
    signals_of_rule = (
        "(SELECT s.id FROM signal s JOIN rule_run r ON r.id = s.rule_run_id WHERE r.rule = ?)"
    )
    for table in ("signal_evidence", "signal_actor", "signal_ai_review"):
        con.execute(f"DELETE FROM {table} WHERE signal_id IN {signals_of_rule}", (rule,))  # noqa: S608
    con.execute(
        "DELETE FROM signal WHERE rule_run_id IN (SELECT id FROM rule_run WHERE rule = ?)", (rule,)
    )
    con.execute("DELETE FROM rule_run WHERE rule = ?", (rule,))
    con.commit()
