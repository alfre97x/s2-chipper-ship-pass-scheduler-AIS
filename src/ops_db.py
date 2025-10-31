import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta


DEFAULT_DB_PATH = Path("ops.db")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Base tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ais_observations (
                id TEXT PRIMARY KEY,            -- deterministic key e.g. mmsi|captured_time|tile_id
                mmsi TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                tile_id TEXT,
                captured_time TEXT,
                overpass_time TEXT,
                source TEXT,
                provider TEXT,
                status TEXT DEFAULT 'captured',
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chip_jobs (
                chip_id TEXT PRIMARY KEY,
                mmsi TEXT,
                scene_id TEXT,
                status TEXT NOT NULL,           -- started|done|failed
                error TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Online schema migrations (best-effort; ignore if columns already exist)
        try:
            cur.execute("ALTER TABLE ais_observations ADD COLUMN retry_count INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE ais_observations ADD COLUMN last_error TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE ais_observations ADD COLUMN last_attempt_at TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE ais_observations ADD COLUMN next_retry_at TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE chip_jobs ADD COLUMN last_attempt_at TEXT")
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()


def upsert_ais_rows(rows: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Upsert AIS rows into ais_observations. Each row should include:
      mmsi, lat, lon, tile_id, captured_time, overpass_time, source, provider
    The id is computed as f"{mmsi}|{captured_time or overpass_time}|{tile_id or ''}".
    Preserves existing status/created_at/retry/backoff fields when present.
    """
    conn = sqlite3.connect(str(db_path))
    inserted = 0
    try:
        cur = conn.cursor()
        for r in rows:
            mmsi = str(r.get("mmsi"))
            lat = float(r.get("lat"))
            lon = float(r.get("lon"))
            tile_id = (r.get("tile_id") or "") and str(r.get("tile_id"))
            captured_time = r.get("captured_time")
            overpass_time = r.get("overpass_time")
            source = r.get("source") or ""
            provider = r.get("provider") or ""
            key_time = captured_time or overpass_time or ""
            doc_id = f"{mmsi}|{key_time}|{tile_id}"
            cur.execute(
                """
                INSERT OR REPLACE INTO ais_observations
                (id, mmsi, lat, lon, tile_id, captured_time, overpass_time, source, provider,
                 status, created_at, retry_count, last_error, last_attempt_at, next_retry_at)
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT status FROM ais_observations WHERE id=?), 'captured'),
                    COALESCE((SELECT created_at FROM ais_observations WHERE id=?), ?),
                    COALESCE((SELECT retry_count FROM ais_observations WHERE id=?), 0),
                    COALESCE((SELECT last_error FROM ais_observations WHERE id=?), NULL),
                    COALESCE((SELECT last_attempt_at FROM ais_observations WHERE id=?), NULL),
                    COALESCE((SELECT next_retry_at FROM ais_observations WHERE id=?), NULL)
                )
                """,
                (
                    doc_id,
                    mmsi,
                    lat,
                    lon,
                    tile_id,
                    captured_time,
                    overpass_time,
                    source,
                    provider,
                    doc_id,
                    doc_id,
                    utcnow_iso(),
                    doc_id,
                    doc_id,
                    doc_id,
                    doc_id,
                ),
            )
            inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()


def upsert_chip_job(
    chip_id: str,
    mmsi: str,
    scene_id: Optional[str],
    status: str,
    error: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """
    Upsert chip job status and bump last_attempt_at.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        now = utcnow_iso()
        cur.execute(
            """
            INSERT INTO chip_jobs (chip_id, mmsi, scene_id, status, error, created_at, last_attempt_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chip_id) DO UPDATE SET
              scene_id=excluded.scene_id,
              status=excluded.status,
              error=excluded.error,
              last_attempt_at=excluded.last_attempt_at
            """,
            (chip_id, mmsi, scene_id, status, error, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def set_ais_status(doc_id: str, status: str, error: Optional[str] = None, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Update ais_observations status and last_error; set last_attempt_at=now.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE ais_observations
            SET status=?, last_error=?, last_attempt_at=?
            WHERE id=?
            """,
            (status, error, utcnow_iso(), doc_id),
        )
        conn.commit()
    finally:
        conn.close()


def increment_retry(
    doc_id: str,
    initial_backoff_min: int = 5,
    max_backoff_hours: int = 24,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """
    Increment retry_count and compute next_retry_at = now + min(initial*2^retry_count, max_backoff_hours).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT retry_count FROM ais_observations WHERE id=?", (doc_id,))
        row = cur.fetchone()
        retry_count = int(row[0]) if row and row[0] is not None else 0
        next_count = retry_count + 1
        minutes = min(initial_backoff_min * (2 ** retry_count), max_backoff_hours * 60)
        next_at = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        cur.execute(
            """
            UPDATE ais_observations
            SET retry_count=?, next_retry_at=?, last_attempt_at=?
            WHERE id=?
            """,
            (next_count, next_at, utcnow_iso(), doc_id),
        )
        conn.commit()
    finally:
        conn.close()


def due_ais(limit: int = 50, now_iso: Optional[str] = None, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Return failed AIS observations due for retry (status='failed' and next_retry_at <= now or NULL).
    """
    if now_iso is None:
        now_iso = utcnow_iso()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, mmsi, lat, lon, tile_id, captured_time, overpass_time
            FROM ais_observations
            WHERE status='failed' AND (next_retry_at IS NULL OR next_retry_at <= ?)
            ORDER BY COALESCE(next_retry_at, '1970-01-01T00:00:00Z') ASC, created_at ASC
            LIMIT ?
            """,
            (now_iso, int(limit)),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
