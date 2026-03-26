import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class RatingEntry:
    prefix: str
    suffix: str
    suggestion: str
    rating: Optional[int]
    suggestion_type: str
    accepted: bool
    timestamp: str


class RatingRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rating (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                prefix           TEXT    NOT NULL,
                suffix           TEXT    NOT NULL DEFAULT '',
                suggestion       TEXT    NOT NULL,
                rating           INTEGER CHECK(rating = 0 OR rating = 1 OR rating IS NULL),
                suggestion_type  TEXT    NOT NULL DEFAULT 'realtime',
                accepted         INTEGER NOT NULL DEFAULT 1,
                timestamp        TEXT    NOT NULL
            )
        """)
        # Safe migration: add columns that may be missing in older DBs
        for col, definition in [
            ("suffix",          "TEXT NOT NULL DEFAULT ''"),
            ("suggestion_type", "TEXT NOT NULL DEFAULT 'realtime'"),
            ("accepted",        "INTEGER NOT NULL DEFAULT 1"),
            ("timestamp",       "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE rating ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
        conn.close()

    def save(self, entry: RatingEntry) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO rating
               (prefix, suffix, suggestion, rating, suggestion_type, accepted, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.prefix,
                entry.suffix,
                entry.suggestion,
                entry.rating,
                entry.suggestion_type,
                int(entry.accepted),
                entry.timestamp,
            ),
        )
        conn.commit()
        conn.close()

    def find_all(self) -> list[RatingEntry]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT prefix, suffix, suggestion, rating, suggestion_type, accepted, timestamp FROM rating").fetchall()
        conn.close()
        return [RatingEntry(*row) for row in rows]
