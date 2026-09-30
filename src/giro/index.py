"""data/giro.sqlite: an index of the jobs and a log of what their stages reported.

job.json in each job directory stays the source of truth; the jobs table is a
copy for listing, rebuilt from the job directories when the server starts.
The events table keeps metrics, previews and log lines, so a client that
connects mid-run can draw the curves so far. Progress is not kept: only the
latest value matters, and the server holds that in memory.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    updated REAL NOT NULL,
    status TEXT NOT NULL,
    best INTEGER,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    job TEXT NOT NULL,
    seed INTEGER,
    stage TEXT,
    type TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_attempt ON events (job, seed, type);
"""


class Index:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection, used only from the server's event loop thread.
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def put_job(self, data: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO jobs (id, created, updated, status, best, data) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET updated = excluded.updated, status = excluded.status, "
            "best = excluded.best, data = excluded.data",
            (data["id"], data["created"], time.time(), data["status"], data.get("best"), json.dumps(data)),
        )

    def drop_missing(self, ids: set[str]) -> None:
        """Forget jobs whose directories are gone."""
        for (job_id,) in self.db.execute("SELECT id FROM jobs").fetchall():
            if job_id not in ids:
                self.db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                self.db.execute("DELETE FROM events WHERE job = ?", (job_id,))

    def jobs(self) -> list[dict[str, Any]]:
        return [json.loads(d) for (d,) in self.db.execute("SELECT data FROM jobs ORDER BY created DESC")]

    def add_event(self, event: dict[str, Any]) -> None:
        rest = {k: v for k, v in event.items() if k not in ("ts", "job", "seed", "stage", "type")}
        self.db.execute(
            "INSERT INTO events (ts, job, seed, stage, type, data) VALUES (?, ?, ?, ?, ?, ?)",
            (event["ts"], event["job"], event.get("seed"), event.get("stage"), event["type"],
             json.dumps(rest, default=str)),
        )

    def events(self, job: str, seed: int | None = None, types: list[str] | None = None,
               since: float = 0.0, limit: int = 5000) -> list[dict[str, Any]]:
        sql, args = "SELECT ts, job, seed, stage, type, data FROM events WHERE job = ? AND ts > ?", [job, since]
        if seed is not None:
            sql += " AND seed = ?"
            args.append(seed)
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            args += types
        # The newest `limit`, returned oldest first.
        sql = f"SELECT * FROM ({sql} ORDER BY id DESC LIMIT ?) ORDER BY ts"
        args.append(limit)
        return [{"ts": ts, "job": j, "seed": s, "stage": st, "type": t} | json.loads(d)
                for ts, j, s, st, t, d in self.db.execute(sql, args)]

    def clear_events(self, job: str, seed: int, stages: list[str]) -> None:
        """Drop what earlier runs of these stages reported (they are about to run again)."""
        self.db.execute(
            f"DELETE FROM events WHERE job = ? AND seed = ? AND stage IN ({','.join('?' * len(stages))})",
            [job, seed, *stages],
        )
