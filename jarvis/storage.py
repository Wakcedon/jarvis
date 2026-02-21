from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TodoItem:
    id: int
    text: str
    done: bool
    created_ts: float


@dataclass(frozen=True)
class ReminderItem:
    id: int
    due_ts: float
    text: str
    fired: bool
    created_ts: float


class SQLiteStorage:
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_ts REAL NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS todo (
                  id INTEGER PRIMARY KEY,
                  text TEXT NOT NULL,
                  done INTEGER NOT NULL DEFAULT 0,
                  created_ts REAL NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                  id INTEGER PRIMARY KEY,
                  due_ts REAL NOT NULL,
                  text TEXT NOT NULL,
                  fired INTEGER NOT NULL DEFAULT 0,
                  created_ts REAL NOT NULL
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_ts)")
            c.commit()

    # --- Memory ---
    def mem_get(self, key: str) -> str | None:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM memory WHERE key = ?", (key,))
            row = cur.fetchone()
            return str(row[0]) if row else None

    def mem_set(self, key: str, value: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory(key, value, updated_ts) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
                (key, value, now),
            )
            self._conn.commit()

    def mem_clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory")
            self._conn.commit()

    # --- Todo ---
    def todo_list(self) -> list[TodoItem]:
        with self._lock:
            cur = self._conn.execute("SELECT id, text, done, created_ts FROM todo ORDER BY id ASC")
            rows = cur.fetchall()
        return [TodoItem(id=int(r[0]), text=str(r[1]), done=bool(r[2]), created_ts=float(r[3])) for r in rows]

    def todo_add(self, text: str) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM todo")
            new_id = int(cur.fetchone()[0])
            self._conn.execute(
                "INSERT INTO todo(id, text, done, created_ts) VALUES(?, ?, 0, ?)",
                (new_id, text, now),
            )
            self._conn.commit()
        return new_id

    def todo_mark_done(self, task_id: int, *, done: bool) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE todo SET done = ? WHERE id = ?", (1 if done else 0, int(task_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def todo_delete(self, task_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM todo WHERE id = ?", (int(task_id),))
            self._conn.commit()
            return cur.rowcount > 0

    # --- Reminders ---
    def reminders_list(self) -> list[ReminderItem]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, due_ts, text, fired, created_ts FROM reminders ORDER BY due_ts ASC, id ASC"
            )
            rows = cur.fetchall()
        return [
            ReminderItem(id=int(r[0]), due_ts=float(r[1]), text=str(r[2]), fired=bool(r[3]), created_ts=float(r[4]))
            for r in rows
        ]

    def reminders_add(self, *, due_ts: float, text: str) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM reminders")
            new_id = int(cur.fetchone()[0])
            self._conn.execute(
                "INSERT INTO reminders(id, due_ts, text, fired, created_ts) VALUES(?, ?, ?, 0, ?)",
                (new_id, float(due_ts), text, now),
            )
            self._conn.commit()
        return new_id

    def reminders_delete(self, rid: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM reminders WHERE id = ?", (int(rid),))
            self._conn.commit()
            return cur.rowcount > 0

    def reminders_due_and_mark_fired(self, *, now_ts: float) -> list[ReminderItem]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, due_ts, text, fired, created_ts FROM reminders WHERE fired = 0 AND due_ts <= ? ORDER BY due_ts ASC",
                (float(now_ts),),
            )
            rows = cur.fetchall()
            ids = [int(r[0]) for r in rows]
            if ids:
                self._conn.executemany("UPDATE reminders SET fired = 1 WHERE id = ?", [(i,) for i in ids])
                self._conn.commit()

        return [
            ReminderItem(id=int(r[0]), due_ts=float(r[1]), text=str(r[2]), fired=bool(r[3]), created_ts=float(r[4]))
            for r in rows
        ]
