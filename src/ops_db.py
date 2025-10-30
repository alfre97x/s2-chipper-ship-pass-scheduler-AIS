import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, Optional
from datetime import datetime, timezone


DEFAULT_DB_PATH = Path("ops.db")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
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
        conn.commit()
    finally:
        conn.close()


def upsert_ais_rows(rows: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Upsert AIS rows into ais_observations. Each row should include:
      mmsi, lat, lon, tile_id, captured_time, overpass_time, source, provider
    The id is computed as f"{mmsi}|{captured_time or overpass_time}|{tile_id or ''}".
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
                (id, mmsi, lat, lon, tile_id, captured_time, overpass_time, source, provider, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT status FROM ais_observations WHERE id=?), 'captured'), COALESCE((SELECT created_at FROM ais_observations WHERE id=?), ?))
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
                ),
            )
            inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()


def upsert_chip_job(chip_id: str, mmsi: str, scene_id: Optional[str], status: str, error: Optional[str] = None, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Upsert chip job status.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chip_jobs (chip_id, mmsi, scene_id, status, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chip_id) DO UPDATE SET
              scene_id=excluded.scene_id,
              status=excluded.status,
              error=excluded.error
            """,
            (chip_id, mmsi, scene_id, status, error, utcnow_iso()),
        )
        conn.commit()
    finally:
        conn.close()
