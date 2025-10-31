import os
import io
import glob
import json
import math
import time
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import yaml
from loguru import logger

# Local pipeline imports
from src.pipeline import (
    load_config,
    run_end_to_end,
    prune_by_ais,
    chip_from_table,
    run_retry_pass,
)
from src.qa_report import generate_qa_report

APP_TITLE = "S2 Ship Chips — Control Panel"

# ---------------------------
# Logging sink into session
# ---------------------------

def _ensure_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = None
    if "worker" not in st.session_state:
        st.session_state.worker = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "log_sink_id" not in st.session_state:
        st.session_state.log_sink_id = None


def _log_sink(msg: str):
    # Append latest log line; cap buffer
    st.session_state.logs.append(msg.rstrip())
    if len(st.session_state.logs) > 400:
        st.session_state.logs = st.session_state.logs[-400:]


def _attach_log_sink():
    # Attach once per session
    if st.session_state.log_sink_id is None:
        sink_id = logger.add(_log_sink, level="INFO")
        st.session_state.log_sink_id = sink_id


# ---------------------------
# Config override utilities
# ---------------------------

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def write_temp_config(overrides: Dict[str, Any], cfg_path: str = "config.yaml") -> str:
    cfg = load_config(cfg_path)
    deep_update(cfg, overrides)
    tmp_path = ".runtime_config.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


# ---------------------------
# Background workers
# ---------------------------

def start_end_to_end_worker(
    cfg_path: str,
    dry_run_ais: bool,
    max_chips: int,
    limit_ships: Optional[int],
    include_swir: bool,
    restrict_to_schedule: bool,
    stop_event: threading.Event
):
    try:
        run_end_to_end(
            cfg_path=cfg_path,
            dry_run_ais=dry_run_ais,
            max_chips=max_chips,
            limit_ships=limit_ships,
            include_swir=include_swir,
            restrict_to_schedule=restrict_to_schedule,
            stop_event=stop_event,
        )
    except Exception as e:
        logger.error(f"End-to-end worker error: {e}")
    finally:
        st.session_state.running = False
        st.session_state.worker = None
        st.session_state.stop_event = None
        logger.info("Worker finished")

def start_continuous_worker(
    cfg_path: str,
    include_swir: bool,
    limit_ships: Optional[int],
    max_chips: Optional[int],
    loop_minutes: int,
    restrict_to_schedule: bool,
    stop_event: threading.Event
):
    """
    Simple continuous loop:
      - runs end-to-end with real AIS (dry_run_ais=False)
      - sleeps loop_minutes
      - repeats until stop_event is set
    """
    try:
        while not stop_event.is_set():
            try:
                run_end_to_end(
                    cfg_path=cfg_path,
                    dry_run_ais=False,
                    max_chips=int(max_chips) if max_chips else 0 or 0,
                    limit_ships=int(limit_ships) if limit_ships else None,
                    include_swir=include_swir,
                    restrict_to_schedule=restrict_to_schedule,
                    stop_event=stop_event,
                )
            except Exception as e:
                logger.error(f"Continuous cycle error: {e}")
            # Sleep in small slices to be responsive to stop
            total = max(1, int(loop_minutes)) * 60
            waited = 0
            while waited < total and not stop_event.is_set():
                time.sleep(1)
                waited += 1
    finally:
        st.session_state.running = False
        st.session_state.worker = None
        st.session_state.stop_event = None
        logger.info("Continuous worker finished")

def start_chip_from_file_worker(
    table_path: str,
    cfg_path: str,
    include_swir: bool,
    limit_ships: Optional[int],
    max_chips: Optional[int],
    stop_event: threading.Event
):
    try:
        chip_from_table(
            table_path=table_path,
            cfg_path=cfg_path,
            include_swir=include_swir,
            limit_ships=limit_ships,
            max_chips=max_chips,
            stop_event=stop_event,
        )
    except Exception as e:
        logger.error(f"Chip-from-file worker error: {e}")
    finally:
        st.session_state.running = False
        st.session_state.worker = None
        st.session_state.stop_event = None
        logger.info("Worker finished")


def start_worker_thread(*args, **kwargs):
    if st.session_state.running:
        st.warning("A job is already running.")
        return
    st.session_state.stop_event = threading.Event()
    worker = threading.Thread(
        target=start_end_to_end_worker,
        kwargs=dict(stop_event=st.session_state.stop_event, **kwargs),
        daemon=True,
    )
    st.session_state.worker = worker
    st.session_state.running = True
    worker.start()


def stop_worker():
    if st.session_state.running and st.session_state.stop_event is not None:
        st.session_state.stop_event.set()
        logger.info("Stop requested")
    else:
        st.info("No running job")


# ---------------------------
# UI helpers
# ---------------------------

def latest_daily_geojson(root: Path) -> Optional[Path]:
    idx_dir = root / "indexes"
    files = sorted(idx_dir.glob("tiles_*.geojson"))
    return files[-1] if files else None


def load_geojson_features(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            fc = json.load(f)
        feats = fc.get("features", [])
        return feats
    except Exception as e:
        logger.warning(f"Failed loading geojson {path}: {e}")
        return []


def quicklook_gallery(root: Path, max_items: int = 12) -> List[Path]:
    # Find latest quicklooks across tree, sorted by mtime desc
    qls = list(root.rglob("*_quicklook.jpg"))
    qls.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return qls[:max_items]


def render_map(feats: List[Dict[str, Any]]):
    # Convert GeoJSON polygon centroids to a DataFrame for mapping
    rows = []
    for f in feats:
        try:
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                # centroid approx as mean of vertices
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                lon = sum(xs) / max(1, len(xs))
                lat = sum(ys) / max(1, len(ys))
                rows.append({
                    "lon": lon,
                    "lat": lat,
                    "chip_id": props.get("chip_id"),
                    "mmsi": props.get("mmsi"),
                    "cloud": props.get("eo_cloud_cover"),
                    "tif": props.get("tif_href"),
                })
        except Exception:
            continue
    if not rows:
        st.info("No footprints to display yet.")
        return
    df = pd.DataFrame(rows)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=150,
        get_fill_color=[255, 0, 0, 120],
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=6)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{chip_id} | {mmsi}"}))


# ---------------------------
# Main App
# ---------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _ensure_state()
    _attach_log_sink()
    st.title(APP_TITLE)

    cfg = load_config("config.yaml")
    root = Path(cfg["storage"]["root"])

    # Sidebar controls
    st.sidebar.header("Run Controls")

    # Mode
    mode = st.sidebar.radio("Mode", options=["Batch", "Continuous"], index=0, horizontal=True)
    include_swir = st.sidebar.checkbox("Include SWIR bands (B11,B12)", value=True)

    # AIS source (real vs synthetic)
    # In real runs we must use real AIS; keep synthetic only for local tests
    if mode == "Continuous":
        st.sidebar.info("Continuous mode uses real AIS (synthetic is disabled).")
        dry_run_ais = False
    else:
        dry_run_ais = st.sidebar.checkbox("Use synthetic AIS (dry-run)", value=False)

    # Limits
    st.sidebar.markdown("Limits")
    limit_ships = st.sidebar.number_input("Max AIS ships to process", min_value=0, max_value=10000, value=50, step=5)
    restrict_to_schedule = st.sidebar.checkbox("Restrict to schedule (predictive)", value=False, help="Collect AIS only within active overpass tiles and windows")
    bind_caps = st.sidebar.checkbox("Bind chips cap to ships cap (1:1)", value=True)
    if bind_caps:
        max_chips = limit_ships if limit_ships and limit_ships > 0 else 10
    else:
        max_chips = st.sidebar.number_input("Max chips this run", min_value=1, max_value=100000, value=10, step=1)

    # Continuous loop interval
    if mode == "Continuous":
        loop_every_min = st.sidebar.number_input("Loop interval (minutes)", min_value=1, max_value=240, value=15, step=1)
    else:
        loop_every_min = None

    # STAC
    st.sidebar.markdown("STAC")
    stac_cloud = st.sidebar.slider("Max cloud cover (%)", min_value=0, max_value=100, value=int(cfg["stac"].get("cloud_cover_max", 30)), step=5)
    stac_window = st.sidebar.slider("Search window (hours, ±)", min_value=1, max_value=240, value=int(cfg["stac"].get("search_window_hours", 24)), step=1)

    # Gate (AIS + sanity)
    st.sidebar.markdown("AIS + Sanity Gate")
    gate_enabled = st.sidebar.checkbox("Enable gate (reject cloudy/land/non-water center)", value=bool(cfg.get("gate", {}).get("enabled", True)))
    center_window_px = st.sidebar.slider("Center window (px)", min_value=16, max_value=128, value=int(cfg.get("gate", {}).get("center_window_px", 64)), step=8)
    max_cloud_center_frac = st.sidebar.slider("Max cloud at center", min_value=0.0, max_value=1.0, value=float(cfg.get("gate", {}).get("max_cloud_center_frac", 0.05)), step=0.01)
    max_land_center_frac = st.sidebar.slider("Max land at center", min_value=0.0, max_value=1.0, value=float(cfg.get("gate", {}).get("max_land_center_frac", 0.2)), step=0.01)
    ndwi_water_thresh = st.sidebar.slider("NDWI water threshold", min_value=-0.5, max_value=0.5, value=float(cfg.get("gate", {}).get("ndwi_water_thresh", 0.05)), step=0.01)
    min_water_center_frac = st.sidebar.slider("Min water at center", min_value=0.0, max_value=1.0, value=float(cfg.get("gate", {}).get("min_water_center_frac", 0.5)), step=0.01)

    # Prune by AIS tolerances
    st.sidebar.markdown("Prune by AIS")
    time_tol_min = st.sidebar.number_input("Time tolerance (min)", min_value=1, max_value=180, value=10, step=1)
    dist_tol_m = st.sidebar.number_input("Distance tolerance (m)", min_value=50, max_value=5000, value=500, step=50)

    # Chip from uploaded AIS file (Excel/CSV)
    st.sidebar.markdown("Chip from AIS file")
    uploaded = st.sidebar.file_uploader("Upload AIS table (.xlsx/.xls/.csv)", type=["xlsx", "xls", "csv"])
    chip_file_clicked = st.sidebar.button("Chip from uploaded file")

    st.sidebar.markdown("---")
    col_b1, col_b2 = st.sidebar.columns(2)
    start_clicked = col_b1.button("Start")
    stop_clicked = col_b2.button("Stop")

    prune_clicked = st.sidebar.button("Prune by AIS (delete mismatches)")

    # Ops controls
    st.sidebar.markdown("---")
    st.sidebar.header("Ops Controls")
    ops_retry_limit = st.sidebar.number_input(
        "Retry pass limit",
        min_value=0,
        max_value=10000,
        value=int(cfg.get("ops", {}).get("retry_batch_limit", 50)),
        step=10,
    )
    ops_retry_clicked = st.sidebar.button("Run Retry Pass")
    qa_report_clicked = st.sidebar.button("Generate QA Report")

    # Handle actions
    # Write overrides to a temp config so pipeline picks them up
    overrides = {
        "stac": {
            "cloud_cover_max": int(stac_cloud),
            "search_window_hours": int(stac_window),
        },
        "gate": {
            "enabled": bool(gate_enabled),
            "center_window_px": int(center_window_px),
            "max_cloud_center_frac": float(max_cloud_center_frac),
            "max_land_center_frac": float(max_land_center_frac),
            "ndwi_water_thresh": float(ndwi_water_thresh),
            "min_water_center_frac": float(min_water_center_frac),
        }
    }
    runtime_cfg = write_temp_config(overrides, cfg_path="config.yaml")

    if start_clicked:
        # Start Batch or Continuous job
        if mode == "Batch":
            start_worker_thread(
                cfg_path=runtime_cfg,
                dry_run_ais=bool(dry_run_ais),
                max_chips=int(max_chips),
                limit_ships=int(limit_ships) if limit_ships and limit_ships > 0 else None,
                include_swir=bool(include_swir),
                restrict_to_schedule=bool(restrict_to_schedule),
            )
        else:
            # Continuous: launch dedicated loop worker (real AIS only)
            if st.session_state.running:
                st.warning("A job is already running.")
            else:
                st.session_state.stop_event = threading.Event()
                worker = threading.Thread(
                    target=start_continuous_worker,
                    kwargs=dict(
                        cfg_path=runtime_cfg,
                        include_swir=bool(include_swir),
                        limit_ships=int(limit_ships) if limit_ships and limit_ships > 0 else None,
                        max_chips=int(max_chips),
                        loop_minutes=int(loop_every_min or 15),
                        restrict_to_schedule=bool(restrict_to_schedule),
                        stop_event=st.session_state.stop_event,
                    ),
                    daemon=True,
                )
                st.session_state.worker = worker
                st.session_state.running = True
                worker.start()
                logger.info("Continuous job started")

    if stop_clicked:
        stop_worker()

    if prune_clicked:
        try:
            prune_by_ais(cfg_path=runtime_cfg, time_tol_min=int(time_tol_min), dist_tol_m=int(dist_tol_m))
            st.sidebar.success("Prune completed.")
        except Exception as e:
            st.sidebar.error(f"Prune error: {e}")

    # Ops actions
    if ops_retry_clicked:
        try:
            produced = run_retry_pass(cfg_path=runtime_cfg, include_swir=bool(include_swir), limit=int(ops_retry_limit))
            st.sidebar.success(f"Retry pass completed: {produced} items processed")
        except Exception as e:
            st.sidebar.error(f"Retry pass error: {e}")

    if qa_report_clicked:
        try:
            report_path = generate_qa_report(cfg_path=runtime_cfg)
            st.sidebar.success(f"QA report generated: {report_path}")
        except Exception as e:
            st.sidebar.error(f"QA report error: {e}")

    # Chip from uploaded file handler
    if chip_file_clicked:
        if not uploaded:
            st.sidebar.error("Please upload an Excel/CSV file first.")
        else:
            try:
                # Persist uploaded file to a temp path
                tmp_dir = Path(".uploads")
                tmp_dir.mkdir(exist_ok=True)
                tmp_path = tmp_dir / f"ais_uploaded{Path(uploaded.name).suffix}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                if st.session_state.running:
                    st.warning("A job is already running.")
                else:
                    st.session_state.stop_event = threading.Event()
                    worker = threading.Thread(
                        target=start_chip_from_file_worker,
                        kwargs=dict(
                            table_path=str(tmp_path),
                            cfg_path=runtime_cfg,
                            include_swir=bool(include_swir),
                            limit_ships=int(limit_ships) if limit_ships and limit_ships > 0 else None,
                            max_chips=int(max_chips),
                            stop_event=st.session_state.stop_event,
                        ),
                        daemon=True,
                    )
                    st.session_state.worker = worker
                    st.session_state.running = True
                    worker.start()
                    st.sidebar.success(f"Chipping from file started: {tmp_path.name}")
            except Exception as e:
                st.sidebar.error(f"Failed to start chip-from-file: {e}")

    # Main layout: status, logs, previews, map, indexes
    st.subheader("Status")
    if st.session_state.running:
        st.info("Job running...")
    else:
        st.success("Idle")

    # Auto-refresh hint
    st.caption("Tip: Streamlit re-runs on interaction. Use the buttons or tweak a control to refresh the view.")

    # Quick stats
    tiles_parquet = root / cfg["storage"]["layout"]["tiles_index"]
    stats_cols = st.columns(3)
    with stats_cols[0]:
        st.metric("tiles.parquet exists", "Yes" if tiles_parquet.exists() else "No")
    with stats_cols[1]:
        if tiles_parquet.exists():
            try:
                df_tiles = pd.read_parquet(tiles_parquet)
                st.metric("Total chips", f"{len(df_tiles)}")
            except Exception:
                st.metric("Total chips", "N/A")
        else:
            st.metric("Total chips", "0")
    with stats_cols[2]:
        st.metric("Dataset root", str(root))

    # Logs
    with st.expander("Logs", expanded=True):
        st.text("\n".join(st.session_state.logs[-200:]))

    # Previews and map
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Quicklook Gallery")
        qls = quicklook_gallery(root, max_items=12)
        if qls:
            cols = st.columns(3)
            for i, p in enumerate(qls):
                with cols[i % 3]:
                    st.image(str(p), caption=p.name, use_column_width=True)
        else:
            st.write("No quicklooks yet.")

    with col_right:
        st.subheader("Footprints Map")
        latest_fc = latest_daily_geojson(root)
        if latest_fc and latest_fc.exists():
            feats = load_geojson_features(latest_fc)
            render_map(feats)
        else:
            st.write("No daily FeatureCollection yet.")

    # Index snapshot
    st.subheader("Index Snapshot (tiles.parquet head)")
    if tiles_parquet.exists():
        try:
            df_head = pd.read_parquet(tiles_parquet).head(50)
            st.dataframe(df_head, use_container_width=True)
        except Exception as e:
            st.write(f"Failed to read tiles.parquet: {e}")
    else:
        st.write("No tiles.parquet yet.")


if __name__ == "__main__":
    main()
