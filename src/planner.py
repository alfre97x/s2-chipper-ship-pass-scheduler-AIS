import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from loguru import logger


def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_esa_csv(url: str, dest_dir: Path) -> Optional[Path]:
    """
    Download an ESA Acquisition Plan CSV (or CSV exported from KML) from the given URL.
    Saves to dest_dir as esa_acquisition_plan_YYYYMMDD.csv.
    Returns the saved path, or None on failure.
    """
    try:
        ensure_dir(dest_dir)
        out_path = dest_dir / f"esa_acquisition_plan_{utc_today_str()}.csv"
        logger.info(f"Downloading ESA plan from {url} -> {out_path}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # Basic content-type check; still allow if text
        if "text" not in (r.headers.get("Content-Type") or "") and not url.lower().endswith(".csv"):
            logger.warning(f"Unexpected content type for ESA plan: {r.headers.get('Content-Type')}")
        with open(out_path, "wb") as f:
            f.write(r.content)
        logger.info(f"Saved ESA plan to {out_path} ({len(r.content)} bytes)")
        return out_path
    except Exception as e:
        logger.error(f"ESA plan download failed: {e}")
        return None


def write_versioned_schedule_csv(df, out_csv: Path, cache_dir: Optional[Path] = None) -> None:
    """
    Write the schedule to out_csv, and also write a versioned copy under
    cache_dir/YYYYMMDD/overpass_schedule.csv if cache_dir is provided.
    """
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote overpass schedule: {out_csv} ({len(df)} rows)")
    if cache_dir:
        day_dir = cache_dir / utc_today_str()
        ensure_dir(day_dir)
        ver_path = day_dir / "overpass_schedule.csv"
        df.to_csv(ver_path, index=False)
        logger.info(f"Wrote versioned schedule: {ver_path} ({len(df)} rows))")
