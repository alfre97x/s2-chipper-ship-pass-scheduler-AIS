from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import json
import requests

from loguru import logger

from .pipeline import load_config, fmt_date, utcnow  # reuse config/date utils
from .ops_db import DEFAULT_DB_PATH

HTML_CSS = """
<style>
body { font-family: Arial, sans-serif; margin: 16px; }
h1, h2, h3 { color: #222; }
.section { margin-bottom: 28px; }
.kpi { display: inline-block; margin-right: 24px; padding: 8px 12px; background: #f6f8fa; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; }
th { background: #f0f0f0; }
.small { color: #666; font-size: 12px; }
pre { background: #f7f7f7; padding: 8px; border-radius: 4px; overflow-x: auto; }
hr { border: none; height: 1px; background: #e5e5e5; margin: 16px 0; }
</style>
"""

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _compute_tiles_stats(df_tiles: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    stats["total_chips"] = len(df_tiles)
    stats["by_split"] = (
        df_tiles.groupby("split").size().to_dict()
        if "split" in df_tiles.columns else {}
    )
    # Cloud cover
    if "cloud_cover" in df_tiles.columns:
        cc = df_tiles["cloud_cover"].dropna().astype(float)
        stats["cloud_cover_mean"] = float(cc.mean()) if not cc.empty else None
        stats["cloud_cover_p50"] = float(cc.quantile(0.5)) if not cc.empty else None
        stats["cloud_cover_p90"] = float(cc.quantile(0.9)) if not cc.empty else None
    else:
        stats["cloud_cover_mean"] = None
        stats["cloud_cover_p50"] = None
        stats["cloud_cover_p90"] = None

    # QA fractions
    for col in ["qa_cloud_frac", "qa_water_frac", "qa_land_frac"]:
        if col in df_tiles.columns:
            s = df_tiles[col].dropna().astype(float)
            stats[f"{col}_mean"] = float(s.mean()) if not s.empty else None
        else:
            stats[f"{col}_mean"] = None

    return stats

def _compute_ops_stats(db_path: Path) -> Dict[str, Any]:
    import sqlite3
    stats: Dict[str, Any] = {}
    if not db_path.exists():
        return {
            "has_db": False,
            "counts": {},
            "failure_rate": None,
            "retry_queue": 0,
            "recent_failures": []
        }
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Counts by status
        cur.execute("SELECT status, COUNT(*) AS c FROM ais_observations GROUP BY status")
        rows = cur.fetchall()
        counts = {r["status"] if r["status"] is not None else "NULL": r["c"] for r in rows}
        total = sum(counts.values())
        failed = int(counts.get("failed", 0))
        failure_rate = (failed / total) if total > 0 else None

        # Retry queue (failed with due next_retry_at)
        now_iso = _now_iso()
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM ais_observations
            WHERE status='failed' AND (next_retry_at IS NULL OR next_retry_at <= ?)
            """,
            (now_iso,)
        )
        retry_queue = int(cur.fetchone()["c"])

        # Recent failures
        cur.execute(
            """
            SELECT id, mmsi, lat, lon, tile_id, captured_time, overpass_time, retry_count, last_error, last_attempt_at, next_retry_at
            FROM ais_observations
            WHERE status='failed'
            ORDER BY COALESCE(next_retry_at, '1970-01-01T00:00:00Z') ASC, last_attempt_at DESC
            LIMIT 20
            """
        )
        recent = [dict(r) for r in cur.fetchall()]

        stats = {
            "has_db": True,
            "counts": counts,
            "failure_rate": failure_rate,
            "retry_queue": retry_queue,
            "recent_failures": recent
        }
        return stats
    finally:
        conn.close()

def _post_webhook(url: str, payload: Dict[str, Any]) -> None:
    try:
        if not url:
            return
        headers = {"Content-Type": "application/json"}
        requests.post(url, headers=headers, data=json.dumps(payload), timeout=8)
    except Exception as e:
        logger.debug(f"Webhook post failed: {e}")

def generate_qa_report(cfg_path: str = "config.yaml", out_path: Optional[str] = None) -> Path:
    """
    Generate a lightweight HTML QA report combining tiles.parquet metrics and ops.db status.
    Also emits an optional webhook alert if failure_rate exceeds config alerts.threshold_fail_rate.
    Returns the path to the generated HTML file.
    """
    cfg = load_config(cfg_path)
    root = Path(cfg["storage"]["root"])
    tiles_path = root / cfg["storage"]["layout"]["tiles_index"]
    report_tpl = cfg["storage"]["layout"]["qa_report"]  # e.g., QA/reports/{date}.html
    date_str = fmt_date(utcnow())
    out = Path(out_path) if out_path else (root / report_tpl.format(date=date_str))
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load tiles
    df_tiles = None
    tiles_stats: Dict[str, Any] = {}
    if tiles_path.exists():
        try:
            df_tiles = pd.read_parquet(tiles_path)
            tiles_stats = _compute_tiles_stats(df_tiles)
        except Exception as e:
            logger.warning(f"Failed reading tiles index: {e}")
    else:
        tiles_stats = {
            "total_chips": 0,
            "by_split": {},
            "cloud_cover_mean": None,
            "cloud_cover_p50": None,
            "cloud_cover_p90": None,
            "qa_cloud_frac_mean": None,
            "qa_water_frac_mean": None,
            "qa_land_frac_mean": None,
        }

    # Ops DB stats
    db_path = DEFAULT_DB_PATH
    ops_stats = _compute_ops_stats(db_path)

    # Build HTML
    html_parts: List[str] = []
    html_parts.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>QA Report {date_str}</title>{HTML_CSS}</head><body>")
    html_parts.append(f"<h1>Sentinel-2 Ship Chips — QA Report</h1>")
    html_parts.append(f"<div class='small'>Generated: {datetime.now(timezone.utc).isoformat()}</div>")

    # KPI section
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>KPIs</h2>")
    html_parts.append(f"<div class='kpi'><b>Total Chips</b>: {tiles_stats.get('total_chips', 0)}</div>")
    if tiles_stats.get("cloud_cover_mean") is not None:
        html_parts.append(f"<div class='kpi'><b>Cloud mean</b>: {tiles_stats['cloud_cover_mean']:.2f}%</div>")
    if ops_stats.get("has_db"):
        counts = ops_stats.get("counts", {})
        html_parts.append(f"<div class='kpi'><b>AIS Captured</b>: {int(counts.get('captured', 0))}</div>")
        html_parts.append(f"<div class='kpi'><b>AIS Failed</b>: {int(counts.get('failed', 0))}</div>")
        fr = ops_stats.get("failure_rate")
        if fr is not None:
            html_parts.append(f"<div class='kpi'><b>Failure Rate</b>: {fr*100:.1f}%</div>")
        html_parts.append(f"<div class='kpi'><b>Retry Queue</b>: {int(ops_stats.get('retry_queue', 0))}</div>")
    html_parts.append("</div>")

    # Split distribution
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>Split distribution</h2>")
    if tiles_stats.get("by_split"):
        by_split = tiles_stats["by_split"]
        rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in by_split.items()])
        html_parts.append(f"<table><thead><tr><th>Split</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>")
    else:
        html_parts.append("<div>No split info available.</div>")
    html_parts.append("</div>")

    # Cloud/QA stats
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>Cloud / QA stats</h2>")
    html_parts.append("<table><tbody>")
    html_parts.append(f"<tr><th>Cloud cover mean</th><td>{tiles_stats.get('cloud_cover_mean')}</td></tr>")
    html_parts.append(f"<tr><th>Cloud cover p50</th><td>{tiles_stats.get('cloud_cover_p50')}</td></tr>")
    html_parts.append(f"<tr><th>Cloud cover p90</th><td>{tiles_stats.get('cloud_cover_p90')}</td></tr>")
    html_parts.append(f"<tr><th>QA cloud frac mean</th><td>{tiles_stats.get('qa_cloud_frac_mean')}</td></tr>")
    html_parts.append(f"<tr><th>QA water frac mean</th><td>{tiles_stats.get('qa_water_frac_mean')}</td></tr>")
    html_parts.append(f"<tr><th>QA land frac mean</th><td>{tiles_stats.get('qa_land_frac_mean')}</td></tr>")
    html_parts.append("</tbody></table>")
    html_parts.append("</div>")

    # Recent failures table
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>Recent failures (top 20)</h2>")
    if ops_stats.get("recent_failures"):
        rows = []
        for r in ops_stats["recent_failures"]:
            rows.append(
                "<tr>" +
                f"<td>{r.get('id','')}</td>" +
                f"<td>{r.get('mmsi','')}</td>" +
                f"<td>{r.get('lat','')}</td>" +
                f"<td>{r.get('lon','')}</td>" +
                f"<td>{r.get('tile_id','')}</td>" +
                f"<td>{r.get('retry_count','')}</td>" +
                f"<td>{r.get('last_error','')}</td>" +
                f"<td>{r.get('last_attempt_at','')}</td>" +
                f"<td>{r.get('next_retry_at','')}</td>" +
                "</tr>"
            )
        html_parts.append(
            "<table><thead><tr>" +
            "<th>id</th><th>mmsi</th><th>lat</th><th>lon</th><th>tile</th><th>retries</th><th>last_error</th><th>last_attempt</th><th>next_retry</th>" +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        html_parts.append("<div>No failures recorded.</div>")
    html_parts.append("</div>")

    # Raw head of tiles.parquet for quick inspection
    if df_tiles is not None and not df_tiles.empty:
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>tiles.parquet head</h2>")
        try:
            html_parts.append(df_tiles.head(20).to_html(index=False))
        except Exception:
            html_parts.append("<div class='small'>Failed to render head()</div>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    out.write_text("".join(html_parts), encoding="utf-8")
    logger.info(f"QA report written: {out}")

    # Optional alert webhook
    alerts_cfg = load_config(cfg_path).get("alerts", {})  # reload to ensure env merged
    thr = alerts_cfg.get("threshold_fail_rate", None)
    url = alerts_cfg.get("webhook_url", "")
    fr = ops_stats.get("failure_rate", None)
    if fr is not None and thr is not None:
        try:
            thr_f = float(thr)
            if fr >= thr_f and url:
                payload = {
                    "event": "qa_alert",
                    "generated_at": _now_iso(),
                    "failure_rate": fr,
                    "retry_queue": ops_stats.get("retry_queue", 0),
                    "counts": ops_stats.get("counts", {}),
                    "report_path": str(out)
                }
                _post_webhook(url, payload)
                logger.info(f"Alert webhook posted (failure_rate={fr:.3f} ≥ {thr_f:.3f})")
        except Exception as e:
            logger.debug(f"Alert evaluation failed: {e}")

    return out
