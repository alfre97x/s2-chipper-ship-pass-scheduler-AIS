import sys
from pathlib import Path
import typer
from loguru import logger
import sqlite3
import json
import os
import pandas as pd
import yaml

from .pipeline import (
    load_config,
    predict_overpasses,
    collect_ais_for_schedule,
    run_end_to_end,
    run_retry_pass,
)
from .qa_report import generate_qa_report
from .stac_publish import publish_stac

app = typer.Typer(help="Sentinel-2 Ship Chips pipeline CLI")


@app.command("predict-overpasses")
def cli_predict_overpasses(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    out_csv: str = typer.Option("plan/overpass_schedule.csv", "--out", "-o", help="Output schedule CSV path"),
    mode: str = typer.Option(None, "--mode", "-m", help="Planner mode: backfill | esa_plan | tle"),
    esa_csv_path: str = typer.Option(None, "--esa-csv", help="Path to ESA Acquisition Plan CSV when using esa_plan"),
):
    """Produce an overpass schedule CSV (supports backfill, ESA plan CSV, and TLE modes)."""
    cfg = load_config(config)
    if mode:
        cfg.setdefault("satellite_tracking", {})["mode"] = mode
    if esa_csv_path:
        cfg.setdefault("satellite_tracking", {})["esa_csv_path"] = esa_csv_path
    out = Path(out_csv)
    predict_overpasses(cfg, out)
    logger.info(f"Overpass schedule written: {out}")


@app.command("collect-ais")
def cli_collect_ais(
    config: str = typer.Option("config.yaml", "--config", "-c"),
    schedule_csv: str = typer.Option("plan/overpass_schedule.csv", "--schedule", "-s"),
    window_min: int = typer.Option(5, "--window-min", "-w"),
    out_dir: str = typer.Option(None, "--out-dir"),
    dry_run: bool = typer.Option(True, help="Generate synthetic AIS if no API key or for testing"),
    restrict_to_schedule: bool = typer.Option(False, "--restrict-to-schedule/--no-restrict-to-schedule", help="Subscribe aisstream only to active overpass tile bounding boxes"),
):
    """Collect AIS around scheduled overpasses (aisstream or synthetic)."""
    cfg = load_config(config)
    out = collect_ais_for_schedule(
        cfg,
        Path(schedule_csv),
        window_min=window_min,
        out_dir=Path(out_dir) if out_dir else None,
        dry_run=dry_run,
        restrict_to_schedule=restrict_to_schedule,
    )
    logger.info(f"AIS parquet: {out}")


@app.command("run")
def cli_run(
    config: str = typer.Option("config.yaml", "--config", "-c"),
    dry_run_ais: bool = typer.Option(True, help="Use synthetic AIS for testing"),
    max_chips: int = typer.Option(5, help="Maximum chips to generate in one run"),
    restrict_to_schedule: bool = typer.Option(False, "--restrict-to-schedule/--no-restrict-to-schedule", help="Collect AIS only within active overpass tile bounding boxes"),
):
    """End-to-end: predict -> collect AIS -> STAC search -> chip tiles."""
    run_end_to_end(cfg_path=config, dry_run_ais=dry_run_ais, max_chips=max_chips, restrict_to_schedule=restrict_to_schedule)


@app.command("qa-report")
def cli_qa_report(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    out: str = typer.Option(None, "--out", "-o", help="Output HTML report path (optional)"),
):
    """Generate a QA HTML report (tiles.parquet metrics + ops.db status)."""
    p = generate_qa_report(cfg_path=config, out_path=out)
    logger.info(f"QA report written: {p}")


@app.command("ops-status")
def cli_ops_status(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    db: str = typer.Option("ops.db", "--db", help="Path to ops SQLite db"),
):
    """Print ops status: ais_observations counts, retry queue size, tiles count."""
    # Tiles count
    try:
        cfg = load_config(config)
        tiles_path = Path(cfg["storage"]["root"]) / cfg["storage"]["layout"]["tiles_index"]
        if tiles_path.exists():
            try:
                df_tiles = pd.read_parquet(tiles_path)
                tiles_count = len(df_tiles)
            except Exception:
                tiles_count = None
        else:
            tiles_count = 0
    except Exception:
        tiles_count = None

    # DB stats
    counts = {}
    retry_queue = None
    try:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*) AS c FROM ais_observations GROUP BY status")
        rows = cur.fetchall()
        counts = {r["status"] if r["status"] is not None else "NULL": r["c"] for r in rows}
        # retry queue (failed and due)
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "SELECT COUNT(*) AS c FROM ais_observations WHERE status='failed' AND (next_retry_at IS NULL OR next_retry_at <= ?)",
            (now_iso,),
        )
        retry_queue = int(cur.fetchone()["c"])
    except Exception as e:
        logger.error(f"Ops status failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    logger.info(f"tiles count: {tiles_count}")
    logger.info(f"ais_observations by status: {counts}")
    logger.info(f"retry queue (due): {retry_queue}")


@app.command("ops-retry")
def cli_ops_retry(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    limit: int = typer.Option(None, "--limit", "-l", help="Max rows to retry (override config ops.retry_batch_limit)"),
    include_swir: bool = typer.Option(True, "--include-swir/--no-include-swir", help="Include SWIR bands when chipping"),
):
    """Run a single retry pass for failed AIS due by backoff."""
    n = run_retry_pass(cfg_path=config, include_swir=include_swir, limit=limit)
    logger.info(f"Retry pass produced {n} chips")


@app.command("publish")
def cli_publish(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    out_manifest: str = typer.Option(None, "--out-manifest", "-o", help="Output manifest path (defaults under dataset_root/manifests)"),
    sample_limit: int = typer.Option(100, "--sample-limit", help="Include up to N sample rows in manifest"),
):
    """
    Create a lightweight dataset manifest (JSON) summarizing tiles index and providing sample entries.
    """
    cfg = load_config(config)
    root = Path(cfg["storage"]["root"])
    tiles_path = root / cfg["storage"]["layout"]["tiles_index"]
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out_manifest) if out_manifest else (manifests_dir / "dataset_manifest.json")

    manifest = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "root": str(root),
        "tiles_index": str(tiles_path),
        "total_chips": 0,
        "by_split": {},
        "samples": [],
        "version": 1
    }

    if tiles_path.exists():
        try:
            df = pd.read_parquet(tiles_path)
            manifest["total_chips"] = int(len(df))
            if "split" in df.columns:
                manifest["by_split"] = df.groupby("split").size().to_dict()
            # sample fields
            cols = [c for c in ["chip_id", "path", "mmsi", "split", "scene_id", "cloud_cover"] if c in df.columns]
            if cols:
                samples = df[cols].head(int(sample_limit)).to_dict(orient="records")
                manifest["samples"] = samples
        except Exception as e:
            logger.warning(f"Failed to read tiles index: {e}")

    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest written: {out}")


@app.command("publish-stac")
def cli_publish_stac(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    out_dir: str = typer.Option(None, "--out-dir", "-o", help="Output directory for STAC catalog (defaults under dataset_root/stac)"),
    absolute_urls: bool = typer.Option(False, "--absolute-urls/--relative-urls", help="Use absolute URLs in STAC assets"),
    base_url: str = typer.Option(None, "--base-url", help="Base URL for absolute assets (e.g., https://bucket.s3.region.amazonaws.com/prefix)"),
):
    """Publish STAC Catalog/Collection/Items for chips in tiles.parquet."""
    p = publish_stac(cfg_path=config, out_dir=out_dir, absolute_urls=absolute_urls, base_url=base_url)
    logger.info(f"STAC catalog written: {p}")


@app.command("doctor")
def cli_doctor(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    db: str = typer.Option("ops.db", "--db", help="Path to ops SQLite db"),
):
    """Diagnose environment, config, storage, and dependencies."""
    ok = True
    # Config checks
    try:
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key in ["storage", "tiling", "stac"]:
            if key not in cfg:
                ok = False
                logger.error(f"Config missing section: {key}")
        root = Path(cfg["storage"]["root"])
        root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset root OK: {root}")
    except Exception as e:
        ok = False
        logger.error(f"Config load failed: {e}")

    # Dependencies
    try:
        import rasterio  # noqa
        logger.info("rasterio OK")
    except Exception as e:
        ok = False
        logger.error(f"rasterio missing/broken: {e}")
    try:
        import pystac  # noqa
        logger.info("pystac OK")
    except Exception as e:
        logger.warning(f"pystac not available: {e}")

    # DB check
    try:
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version;")
        conn.close()
        logger.info(f"SQLite DB OK: {db}")
    except Exception as e:
        ok = False
        logger.error(f"SQLite DB problem: {e}")

    # Storage layout existence
    try:
        for rel in ["indexes", "QA/reports", "manifests", "stac"]:
            (root / rel).mkdir(parents=True, exist_ok=True)
        logger.info("Storage subdirs present/created (indexes, QA/reports, manifests, stac)")
    except Exception as e:
        ok = False
        logger.error(f"Storage creation failed: {e}")

    if ok:
        logger.info("Doctor checks PASSED")
    else:
        logger.error("Doctor checks FAILED")


@app.command("vacuum")
def cli_vacuum(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    db: str = typer.Option("ops.db", "--db", help="Path to ops SQLite db"),
    remove_empty_dirs: bool = typer.Option(True, "--rm-empty/--keep-empty", help="Remove empty directories under dataset_root"),
):
    """Compact ops.db and optionally remove empty directories."""
    # VACUUM DB
    try:
        conn = sqlite3.connect(db)
        conn.execute("VACUUM;")
        conn.close()
        logger.info("ops.db VACUUM done")
    except Exception as e:
        logger.error(f"VACUUM failed: {e}")

    # Remove empty dirs
    try:
        cfg = load_config(config)
        root = Path(cfg["storage"]["root"])
        if remove_empty_dirs and root.exists():
            removed = 0
            for p in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                try:
                    if p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                        removed += 1
                except Exception:
                    pass
            logger.info(f"Removed {removed} empty directories")
    except Exception as e:
        logger.error(f"Empty dir cleanup failed: {e}")


if __name__ == "__main__":
    app()
