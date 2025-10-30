import os
import json
import uuid
import math
import time
import hashlib
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import yaml
from pyproj import Transformer, CRS
from shapely.geometry import Polygon, mapping, Point
from shapely.ops import transform as shp_transform
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.warp import Resampling
from pystac_client import Client


# --------------------------
# Config & environment
# --------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    load_dotenv()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Inject env fallbacks
    cfg.setdefault("storage", {}).setdefault("root", "./dataset_root")
    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fmt_date(d: datetime) -> str:
    return d.strftime("%Y%m%d")


# --------------------------
# Deterministic IDs & splits
# --------------------------

def deterministic_chip_id(mmsi: str, iso_ts: str, lat: float, lon: float) -> str:
    key = f"{mmsi}|{iso_ts}|{round(lat,5)}|{round(lon,5)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def assign_split(chip_id: str, splits: Dict[str, float]) -> str:
    # Hash-based deterministic split
    v = int(hashlib.md5(chip_id.encode("utf-8")).hexdigest(), 16) / (1 << 128)
    acc = 0.0
    for name, frac in [("train", splits.get("train", 0.8)),
                       ("val", splits.get("val", 0.1)),
                       ("test", splits.get("test", 0.1))]:
        acc += frac
        if v <= acc:
            return name
    return "train"


# --------------------------
# File layout helpers
# --------------------------

def layout_paths(cfg: Dict[str, Any], split: str, mmsi: str, date_str: str, chip_id: str) -> Dict[str, Path]:
    root = Path(cfg["storage"]["root"])
    layout = cfg["storage"]["layout"]
    # Build concrete relative paths
    def fill(pat: str) -> Path:
        return root / pat.format(split=split, mmsi=mmsi, date=date_str, chip_id=chip_id)
    return {
        "tif": fill(layout["chips"]),
        "geojson": fill(layout["chip_geojson"]),
        "meta": fill(layout["chip_meta"]),
        "quicklook": fill(layout["chip_quicklook"]),
        "daily_geojson": root / layout["daily_tiles_geojson"].format(date=date_str),
        "tiles_index": root / layout["tiles_index"],
        "ais_index": root / layout["ais_index"],
        "qa_report": root / layout["qa_report"].format(date=date_str),
    }


# --------------------------
# Overpass prediction (MVP)
# --------------------------

def predict_overpasses(cfg: Dict[str, Any], out_csv: Path) -> None:
    """
    Minimal MVP: emit a backfilled schedule (past N days at 10:30 UTC)
    so STAC search finds available scenes during testing. Upgrade to ESA/TLE for production.
    """
    tiles = cfg["satellite_tracking"].get("tiles", ["T30SUJ"])
    window_min = int(cfg["satellite_tracking"].get("overpass_window_minutes", 5))
    days_back = int(cfg["satellite_tracking"].get("prediction_days", 5))
    rows = []
    now = utcnow()
    for day in range(1, days_back + 1):
        t = (now - timedelta(days=day)).replace(hour=10, minute=30, second=0, microsecond=0)
        for tile in tiles:
            rows.append({
                "tile_id": tile,
                "satellite": "S2X",
                "start_time_utc": (t - timedelta(minutes=window_min)).isoformat(),
                "end_time_utc": (t + timedelta(minutes=window_min)).isoformat()
            })
    df = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote overpass schedule: {out_csv} ({len(df)} rows)")


# --------------------------
# AIS collection (aisstream or dry-run)
# --------------------------

def collect_ais_for_schedule(cfg: Dict[str, Any], schedule_csv: Path, window_min: int = 5, out_dir: Optional[Path] = None, dry_run: bool = False) -> Path:
    """
    Collect AIS around scheduled overpasses. If AISSTREAM_API_KEY is missing or dry_run=True,
    generate synthetic AIS around a test AOI.
    """
    api_key = os.getenv("AISSTREAM_API_KEY")
    date_str = fmt_date(utcnow())
    out_dir = out_dir or Path("ais") / date_str
    ensure_dir(out_dir)

    schedule = pd.read_csv(schedule_csv, parse_dates=["start_time_utc", "end_time_utc"])
    # Keep only windows that intersect now (MVP) or just the first window for demo
    if schedule.empty:
        raise RuntimeError("Empty overpass schedule")
    # For simplicity, take the first row for demo collection
    row = schedule.iloc[0]
    start_t = pd.to_datetime(row["start_time_utc"], utc=True)
    end_t = pd.to_datetime(row["end_time_utc"], utc=True)
    tile_id = row["tile_id"]

    if dry_run or not api_key:
        # Generate synthetic AIS around Gibraltar to exercise the pipeline
        center_lat, center_lon = 36.1, -5.4
        n = 50
        ships = []
        for i in range(n):
            mmsi = str(200000000 + i)
            lat = center_lat + random.uniform(-0.3, 0.3)
            lon = center_lon + random.uniform(-0.3, 0.3)
            ts = start_t + (end_t - start_t) * random.random()
            ships.append({
                "mmsi": mmsi,
                "lat": lat,
                "lon": lon,
                "tile_id": tile_id,
                "overpass_time": start_t.isoformat(),
                "captured_time": ts.isoformat(),
                "source": "predictive",
                "provider": "dry_run"
            })
        df = pd.DataFrame(ships)
        out_path = out_dir / f"ais_overpass_{tile_id}.parquet"
        df.to_parquet(out_path, index=False)
        # Also write Excel snapshot for resilience
        try:
            out_xlsx = out_path.with_suffix(".xlsx")
            df.to_excel(out_xlsx, index=False)
        except Exception as e:
            logger.warning(f"Failed writing Excel AIS snapshot: {e}")
        logger.info(f"Wrote synthetic AIS sample: {out_path} ({len(df)} rows)")
        return out_path
    else:
        # Real-time aisstream listener (MVP: not long-running; short sample)
        import asyncio
        import websockets
        import json as pyjson

        async def listen():
            url = "wss://stream.aisstream.io/v0/stream"
            ships: Dict[str, Dict[str, Any]] = {}
            bbox = []  # empty == global
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    await ws.send(pyjson.dumps({"APIKey": api_key, "BoundingBoxes": bbox}))
                    t_end = time.time() + (window_min * 60)
                    while time.time() < t_end:
                        msg = pyjson.loads(await ws.recv())
                        ais = msg.get("Message", {}).get("PositionReport", {})
                        if not ais:
                            continue
                        mmsi = str(msg.get("MetaData", {}).get("MMSI"))
                        lat = ais.get("Latitude")
                        lon = ais.get("Longitude")
                        ts = ais.get("TimeUTC") or utcnow().isoformat()
                        if mmsi and lat and lon:
                            ships[mmsi] = {
                                "mmsi": mmsi,
                                "lat": float(lat),
                                "lon": float(lon),
                                "tile_id": tile_id,
                                "overpass_time": start_t.isoformat(),
                                "captured_time": ts,
                                "source": "predictive",
                                "provider": "aisstream"
                            }
            except Exception as e:
                logger.error(f"AIS stream error: {e}")

            df = pd.DataFrame(list(ships.values()))
            out_path = out_dir / f"ais_overpass_{tile_id}.parquet"
            if not df.empty:
                df.to_parquet(out_path, index=False)
                # Also write Excel snapshot for resilience
                try:
                    out_xlsx = out_path.with_suffix(".xlsx")
                    df.to_excel(out_xlsx, index=False)
                except Exception as e:
                    logger.warning(f"Failed writing Excel AIS snapshot: {e}")
                logger.info(f"Wrote AIS sample: {out_path} ({len(df)} rows)")
            else:
                # Fallback to synthetic if we failed to collect
                return collect_ais_for_schedule(cfg, schedule_csv, window_min, out_dir, dry_run=True)
            return out_path

        return asyncio.get_event_loop().run_until_complete(listen())


# --------------------------
# STAC search (Earth Search)
# --------------------------

def search_best_scene(cfg: Dict[str, Any], center_lon: float, center_lat: float, overpass_time_iso: str) -> Optional[Dict[str, Any]]:
    stac_cfg = cfg["stac"]
    endpoint = stac_cfg["primary_endpoint"]
    collection = stac_cfg.get("collection", "sentinel-2-l2a")
    cloud_max = stac_cfg.get("cloud_cover_max", 30)
    window_hours = stac_cfg.get("search_window_hours", 24)
    dt_center = pd.to_datetime(overpass_time_iso, utc=True)

    # Fallback strategy: relax constraints progressively (cloud, time window, bbox size)
    cloud_try = []
    for v in [cloud_max, 60, 100]:
        if v not in cloud_try:
            cloud_try.append(v)
    hours_try = []
    for v in [window_hours, 72, 120]:
        if v not in hours_try:
            hours_try.append(v)
    delta_try = [0.05, 0.2, 0.4]

    try:
        client = Client.open(endpoint, ignore_conformance=True)
        for cc in cloud_try:
            for hrs in hours_try:
                dt_start = (dt_center - timedelta(hours=hrs)).isoformat()
                dt_end = (dt_center + timedelta(hours=hrs)).isoformat()
                for delta in delta_try:
                    bbox = [center_lon - delta, center_lat - delta, center_lon + delta, center_lat + delta]
                    search = client.search(
                        collections=[collection],
                        bbox=bbox,
                        datetime=f"{dt_start}/{dt_end}",
                        query={"eo:cloud_cover": {"lte": cc}},
                        max_items=stac_cfg.get("max_items_per_overpass", 5),
                    )
                    # Use non-deprecated iterator
                    items = list(search.items())
                    if items:
                        # Pick closest in time to overpass center
                        def dt_of(item):
                            return abs(pd.to_datetime(item.properties["datetime"], utc=True) - dt_center)
                        best = sorted(items, key=dt_of)[0]
                        assets = {k: v.href for k, v in best.assets.items()}
                        logger.info(f"Selected STAC item {best.id} (cloud≤{cc}, window±{hrs}h, delta={delta})")
                        return {
                            "id": best.id,
                            "collection": best.collection_id,
                            "datetime": pd.to_datetime(best.properties["datetime"], utc=True).isoformat(),
                            "bbox": best.bbox,
                            "assets": assets,
                            "cloud_cover": best.properties.get("eo:cloud_cover", None)
                        }
        logger.warning("No STAC items found after relaxing constraints (cloud/time/bbox)")
        return None
    except Exception as e:
        logger.error(f"STAC search failed: {e}")
        return None


# --------------------------
# Raster chipping
# --------------------------

def _resolve_asset_key(assets: Dict[str, Any], desired: str) -> Optional[str]:
    keys = list(assets.keys())
    name = desired.upper()

    # Provider alias maps (Earth Search naming)
    alias_map = {
        "B02": ["blue", "coastal"],           # 10m
        "B03": ["green"],                     # 10m
        "B04": ["red"],                       # 10m
        "B08": ["nir", "nir08"],              # 10m
        "B11": ["swir16"],                    # 20m
        "B12": ["swir22"],                    # 20m
        "SCL": ["scl", "scl_20m", "classification", "scene_classification"],
    }

    # Prefer COG/GeoTIFF keys over JP2 when both exist
    def best_from(candidates: List[str]) -> Optional[str]:
        # strict matches in given order
        for c in candidates:
            if c in assets:
                return c
            # also consider hyphenated variants like "blue-jp2"
            if f"{c}-jp2" in assets:
                # keep as fallback, but only if nothing else is present
                pass
        # if no straight match, allow jp2 variants explicitly
        for c in candidates:
            jp2 = f"{c}-jp2"
            if jp2 in assets:
                return jp2
        return None

    # 1) Alias resolution
    if name in alias_map:
        res = best_from(alias_map[name])
        if res:
            return res

    # 2) Generic candidate expansion
    candidates: List[str] = [desired, desired.upper(), desired.lower()]
    if name.startswith("B"):  # spectral bands
        band = name
        candidates.extend([
            f"{band}_10m", f"{band}_20m", f"{band}_60m",
            f"{band.lower()}_10m", f"{band.lower()}_20m", f"{band.lower()}_60m",
        ])
    if name == "SCL":
        candidates.extend(["SCL_20m", "scl_20m", "classification", "scene_classification"])

    res = best_from([c if isinstance(c, str) else str(c) for c in candidates])
    if res:
        return res

    # 3) Fuzzy: any key that starts with the desired token (case-insensitive)
    for k in keys:
        if k.upper().startswith(name):
            return k
    return None

def _open_asset(assets: Dict[str, str], key: str):
    resolved = _resolve_asset_key(assets, key)
    if resolved is None:
        raise FileNotFoundError(f"Asset {key} not found in STAC item; available keys: {list(assets.keys())}")
    href = assets[resolved]
    return rasterio.open(href)


def _read_window_scaled(ds, window: Window, out_shape: Tuple[int, int], resampling: Resampling, boundless: bool = False) -> np.ndarray:
    arr = ds.read(1, window=window, out_shape=out_shape, resampling=resampling, boundless=boundless, fill_value=0)
    # Convert to float32 reflectance in [0,1]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 10000.0
    else:
        arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _gate_ship_present(stack: np.ndarray, scl: Optional[np.ndarray], bands_core: List[str], gate_cfg: Dict[str, Any]) -> bool:
    try:
        if not gate_cfg.get("enabled", False):
            return True
        C, H, W = stack.shape
        win = int(gate_cfg.get("center_window_px", 64))
        win = max(8, min(win, min(H, W)))
        r0 = H // 2 - win // 2
        c0 = W // 2 - win // 2
        r1 = r0 + win
        c1 = c0 + win

        # Cloud/Land fractions from SCL if available
        max_cloud = float(gate_cfg.get("max_cloud_center_frac", 0.05))
        max_land = float(gate_cfg.get("max_land_center_frac", 0.2))
        if scl is not None:
            scl_win = scl[r0:r1, c0:c1]
            cloud_center = np.isin(scl_win, [7, 8, 9, 10]).mean()
            land_center = np.isin(scl_win, [4, 5]).mean()
            if cloud_center > max_cloud or land_center > max_land:
                return False

        # NDWI-based water presence
        ndwi_water_thresh = float(gate_cfg.get("ndwi_water_thresh", 0.05))
        min_water = float(gate_cfg.get("min_water_center_frac", 0.5))
        if "B03" in bands_core and "B08" in bands_core:
            b03_idx = bands_core.index("B03")
            b08_idx = bands_core.index("B08")
            g = stack[b03_idx, r0:r1, c0:c1]
            nir = stack[b08_idx, r0:r1, c0:c1]
            ndwi = (g - nir) / (g + nir + 1e-6)
            water_frac = (ndwi > ndwi_water_thresh).mean()
            if water_frac < min_water:
                return False
        # If missing bands, accept by default
        return True
    except Exception:
        # Defensive: if gate errors, do not block pipeline
        return True

def chip_one(
    cfg: Dict[str, Any],
    mmsi: str,
    lat: float,
    lon: float,
    overpass_time_iso: str,
    stac_item: Dict[str, Any],
    include_swir: bool = True
) -> Optional[Dict[str, Any]]:
    tiling = cfg["tiling"]
    chip_px = int(tiling.get("chip_size_px", 256))
    half = chip_px // 2
    bands_core = tiling.get("bands_core", ["B02", "B03", "B04", "B08"])
    bands_swir = tiling.get("bands_swir", ["B11", "B12"]) if include_swir else []
    want_swir = include_swir and len(bands_swir) == 2

    assets = stac_item["assets"]
    # Open a 10m band for geometry (B02)
    with _open_asset(assets, bands_core[0]) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        # Project point to ref CRS
        transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        col, row = ~ref_transform * (x, y)
        col = int(round(col))
        row = int(round(row))
        window = Window(col_off=col - half, row_off=row - half, width=chip_px, height=chip_px)
        # Allow boundless reads; if window exceeds dataset, missing areas will be zero-filled
        # Compute chip transform
        chip_transform = rasterio.windows.transform(window, ref_transform)

        # Read core bands
        core_arrays = []
        for b in bands_core:
            with _open_asset(assets, b) as ds:
                arr = _read_window_scaled(ds, window, (chip_px, chip_px), Resampling.bilinear, boundless=True)
                core_arrays.append(arr)
        stack = np.stack(core_arrays, axis=0)  # (C,H,W)

        # Optional SWIR resampled to 10m
        if want_swir:
            for b in bands_swir:
                with _open_asset(assets, b) as ds:
                    arr = _read_window_scaled(ds, window, (chip_px, chip_px), Resampling.bilinear, boundless=True)
                    stack = np.concatenate([stack, arr[None, ...]], axis=0)

        # Masks & QA from SCL
        cloud_frac = None
        water_frac = None
        land_frac = None
        try:
            with _open_asset(assets, "SCL") as scl_ds:
                scl = scl_ds.read(1, window=window, out_shape=(chip_px, chip_px), resampling=Resampling.nearest, boundless=True, fill_value=0)
                # SCL classes: clouds {7,8,9,10}, water {6}, land {4,5}
                cloud_mask = np.isin(scl, [7, 8, 9, 10])
                water_mask = scl == 6
                land_mask = np.isin(scl, [4, 5])
                total = chip_px * chip_px
                cloud_frac = float(cloud_mask.sum()) / total
                water_frac = float(water_mask.sum()) / total
                land_frac = float(land_mask.sum()) / total
        except Exception:
            pass

        # AIS+sanity gate (center-window SCL/NDWI checks)
        gate_cfg = cfg.get("gate", {})
        if gate_cfg.get("enabled", False):
            scl_arr = locals().get("scl", None)
            if not _gate_ship_present(stack, scl_arr, bands_core, gate_cfg):
                logger.info("Gate rejected chip (AIS+sanity); skipping")
                return None

        # Prepare output paths
        date_str = fmt_date(pd.to_datetime(stac_item["datetime"], utc=True))
        chip_id = deterministic_chip_id(mmsi, stac_item["datetime"], lat, lon)
        split = assign_split(chip_id, cfg.get("splits", {}))
        paths = layout_paths(cfg, split, mmsi, date_str, chip_id)
        ensure_dir(paths["tif"].parent)

        # Write GeoTIFF (simple GTiff; COG optional later)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": stack.shape[0],
            "height": chip_px,
            "width": chip_px,
            "transform": chip_transform,
            "crs": ref_crs,
            "tiled": True,
            "compress": "deflate",
            "predictor": 2,
        }
        with rasterio.open(paths["tif"], "w", **profile) as dst:
            for i in range(stack.shape[0]):
                dst.write(stack[i, :, :], i + 1)

        # Quicklook (RGB 2-98 stretch)
        try:
            from PIL import Image
            rgb_idx = [bands_core.index("B04"), bands_core.index("B03"), bands_core.index("B02")]  # R,G,B order from core list
            rgb = stack[rgb_idx, :, :]
            lo = np.percentile(rgb, 2)
            hi = np.percentile(rgb, 98)
            rgb8 = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
            rgb8 = (rgb8 * 255).astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(rgb8).save(paths["quicklook"])
        except Exception as e:
            logger.warning(f"Quicklook failed: {e}")

        # Metadata JSON
        meta = {
            "chip_id": chip_id,
            "split": split,
            "mmsi": mmsi,
            "center": {"lat": lat, "lon": lon},
            "overpass_time": overpass_time_iso,
            "scene": {
                "id": stac_item["id"],
                "collection": stac_item["collection"],
                "datetime": stac_item["datetime"],
                "cloud_cover": stac_item.get("cloud_cover"),
            },
            "bands": {
                "core": bands_core,
                "swir": bands_swir if want_swir else [],
            },
            "paths": {
                "tif": str(paths["tif"]),
                "quicklook": str(paths["quicklook"]),
            },
            "qa": {
                "cloud_frac": cloud_frac,
                "water_frac": water_frac,
                "land_frac": land_frac,
            },
            "crs_epsg": CRS.from_user_input(ref_crs).to_epsg(),
            "transform": list(chip_transform)[:6],
            "gsd_m": cfg["tiling"].get("gsd_m", 10),
            "chip_size_px": chip_px,
            "provider": "EarthSearch",
            "normalization": "reflectance_0_1",
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # GeoJSON footprint
        left, bottom, right, top = rasterio.windows.bounds(window, ref_transform)
        # Corner coordinates in ref CRS
        corners = [(left, bottom), (right, bottom), (right, top), (left, top), (left, bottom)]
        to_wgs84 = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True).transform
        poly_wgs84 = Polygon([to_wgs84(*pt) for pt in corners])
        center_point = Point(lon, lat)
        feature = {
            "type": "Feature",
            "geometry": mapping(poly_wgs84),
            "properties": {
                "chip_id": chip_id,
                "mmsi": mmsi,
                "split": split,
                "center_lon": lon,
                "center_lat": lat,
                "overpass_time": overpass_time_iso,
                "scene_id": stac_item["id"],
                "scene_datetime": stac_item["datetime"],
                "eo_cloud_cover": stac_item.get("cloud_cover"),
                "bands": bands_core + (bands_swir if want_swir else []),
                "crs_epsg": meta["crs_epsg"],
                "transform": meta["transform"],
                "gsd_m": meta["gsd_m"],
                "chip_size_px": chip_px,
                "tif_href": str(paths["tif"]),
                "meta_href": str(paths["meta"]),
                "quicklook_href": str(paths["quicklook"]),
                "qa_cloud_frac": cloud_frac,
                "qa_water_frac": water_frac,
                "qa_land_frac": land_frac,
            }
        }
        with open(paths["geojson"], "w", encoding="utf-8") as f:
            json.dump(feature, f, indent=2)

        # Update daily FeatureCollection and tiles index
        _update_daily_geojson(paths["daily_geojson"], feature)
        _update_tiles_index(paths["tiles_index"], meta, paths["tif"])

        return {
            "paths": paths,
            "meta": meta,
            "feature": feature,
        }


def _update_daily_geojson(daily_fc_path: Path, feature: Dict[str, Any]) -> None:
    ensure_dir(daily_fc_path.parent)
    if daily_fc_path.exists():
        with open(daily_fc_path, "r", encoding="utf-8") as f:
            fc = json.load(f)
        fc["features"].append(feature)
    else:
        fc = {"type": "FeatureCollection", "features": [feature]}
    with open(daily_fc_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)


def _update_tiles_index(tiles_index_path: Path, meta: Dict[str, Any], tif_path: Path) -> None:
    ensure_dir(tiles_index_path.parent)
    row = {
        "chip_id": meta["chip_id"],
        "path": str(tif_path),
        "mmsi": meta["mmsi"],
        "lat": meta["center"]["lat"],
        "lon": meta["center"]["lon"],
        "overpass_time": meta["overpass_time"],
        "scene_id": meta["scene"]["id"],
        "scene_datetime": meta["scene"]["datetime"],
        "cloud_cover": meta["scene"]["cloud_cover"],
        "split": meta["split"],
        "bands": ",".join(meta["bands"]["core"] + meta["bands"]["swir"]),
        "has_swir": len(meta["bands"]["swir"]) > 0,
        "qa_cloud_frac": meta["qa"]["cloud_frac"],
        "qa_water_frac": meta["qa"]["water_frac"],
        "qa_land_frac": meta["qa"]["land_frac"],
    }
    df_new = pd.DataFrame([row])
    if tiles_index_path.exists():
        try:
            df_old = pd.read_parquet(tiles_index_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_parquet(tiles_index_path, index=False)


# --------------------------
# End-to-end convenience
# --------------------------

def run_end_to_end(
    cfg_path: str = "config.yaml",
    dry_run_ais: bool = True,
    max_chips: int = 10,
    limit_ships: Optional[int] = None,
    include_swir: bool = True,
    stop_event: Optional[Any] = None
) -> None:
    """
    Convenience: predict -> collect_ais -> search scene -> chip N samples
    """
    cfg = load_config(cfg_path)
    # 1) Predict overpasses
    schedule_csv = Path("plan") / "overpass_schedule.csv"
    predict_overpasses(cfg, schedule_csv)

    # 2) Collect AIS (synthetic by default)
    ais_path = collect_ais_for_schedule(cfg, schedule_csv, window_min=cfg["satellite_tracking"].get("overpass_window_minutes", 5), dry_run=dry_run_ais)

    # 3) For each AIS point, find best scene and chip
    ais_df = pd.read_parquet(ais_path)
    if ais_df.empty:
        logger.warning("No AIS samples to chip")
        return
    if limit_ships:
        try:
            nlim = int(limit_ships)
            if nlim > 0:
                ais_df = ais_df.head(nlim)
        except Exception:
            pass

    # 4) Chip up to max_chips samples; search a suitable scene per AIS point
    count = 0
    for _, r in ais_df.iterrows():
        if count >= max_chips:
            break

        # Find the best candidate scene for this AIS point/time
        best_item = search_best_scene(cfg, float(r["lon"]), float(r["lat"]), str(r["overpass_time"]))
        if not best_item:
            # No matching item; continue with next AIS point
            continue

        try:
            # Cooperative stop check
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                logger.info("Stop requested; aborting run loop")
                break
            result = chip_one(
                cfg=cfg,
                mmsi=str(r["mmsi"]),
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                overpass_time_iso=str(r["overpass_time"]),
                stac_item=best_item,
                include_swir=include_swir
            )
            if result:
                count += 1
        except Exception as e:
            logger.error(f"Chip failed for MMSI={r['mmsi']}: {e}")

    logger.info(f"Chipped {count} samples (out of {len(ais_df)})")

def chip_from_table(
    table_path: str,
    cfg_path: str = "config.yaml",
    include_swir: bool = True,
    limit_ships: Optional[int] = None,
    max_chips: Optional[int] = None,
    stop_event: Optional[Any] = None
) -> None:
    """
    Chip directly from an Excel/CSV table with columns at least:
      - mmsi (str/int), lat (float), lon (float)
      - overpass_time (ISO8601, optional; if missing, current UTC time is used)
    """
    cfg = load_config(cfg_path)
    p = Path(table_path)
    if not p.exists():
        raise FileNotFoundError(f"AIS table not found: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        ais_df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        ais_df = pd.read_csv(p)
    else:
        raise ValueError("Unsupported table format; use .xlsx, .xls, or .csv")

    # Basic sanity
    for col in ("mmsi", "lat", "lon"):
        if col not in ais_df.columns:
            raise ValueError(f"Required column '{col}' missing from {p.name}")

    if limit_ships:
        try:
            nlim = int(limit_ships)
            if nlim > 0:
                ais_df = ais_df.head(nlim)
        except Exception:
            pass

    produced = 0
    for _, r in ais_df.iterrows():
        try:
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                logger.info("Stop requested during chip_from_table; aborting")
                break
            mmsi = str(r["mmsi"])
            lat = float(r["lat"])
            lon = float(r["lon"])
            # Determine target time
            ovp = r["overpass_time"] if "overpass_time" in r and pd.notna(r["overpass_time"]) else utcnow().isoformat()
            best_item = search_best_scene(cfg, lon, lat, str(ovp))
            if not best_item:
                continue
            res = chip_one(
                cfg=cfg,
                mmsi=mmsi,
                lat=lat,
                lon=lon,
                overpass_time_iso=str(ovp),
                stac_item=best_item,
                include_swir=include_swir
            )
            if res:
                produced += 1
                if max_chips and produced >= int(max_chips):
                    break
        except Exception as e:
            logger.error(f"Chip-from-table failed for MMSI={r.get('mmsi')}: {e}")

    logger.info(f"Chip-from-table produced {produced} chips from {len(ais_df)} AIS rows")

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    try:
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    except Exception:
        return float("nan")

def prune_by_ais(cfg_path: str = "config.yaml", time_tol_min: int = 10, dist_tol_m: int = 500) -> None:
    """
    Prune existing chips that do not match any AIS observation within time/dist tolerances.
    Deletes artifacts and updates tiles.parquet and daily FeatureCollections.
    """
    cfg = load_config(cfg_path)
    root = Path(cfg["storage"]["root"])
    tiles_index = root / cfg["storage"]["layout"]["tiles_index"]
    if not tiles_index.exists():
        logger.warning(f"No tiles index found at {tiles_index}")
        return
    tiles = pd.read_parquet(tiles_index)
    if tiles.empty:
        logger.info("No tiles to prune")
        return

    # Load AIS snapshots from ais/YYYYMMDD/*.parquet
    ais_root = Path("ais")
    ais_files = list(ais_root.rglob("*.parquet"))
    if not ais_files:
        logger.warning("No AIS parquet files found under ais/")
        return
    ais_frames = [pd.read_parquet(p).assign(_src=str(p)) for p in ais_files]
    ais = pd.concat(ais_frames, ignore_index=True)

    # Coerce types and times
    tiles["scene_datetime"] = pd.to_datetime(tiles["scene_datetime"], utc=True, errors="coerce")
    if "captured_time" in ais.columns:
        ais["captured_time"] = pd.to_datetime(ais["captured_time"], utc=True, errors="coerce")
    else:
        if "overpass_time" in ais.columns:
            ais["captured_time"] = pd.to_datetime(ais["overpass_time"], utc=True, errors="coerce")
        else:
            logger.warning("AIS data lacks time fields; aborting prune")
            return
    ais["mmsi"] = ais["mmsi"].astype(str)
    tiles["mmsi"] = tiles["mmsi"].astype(str)

    to_delete: List[str] = []
    for row in tiles.itertuples(index=False):
        sub = ais[ais["mmsi"] == row.mmsi]
        if sub.empty or pd.isna(row.scene_datetime):
            to_delete.append(row.chip_id)
            continue
        deltas = (sub["captured_time"] - row.scene_datetime).abs().dt.total_seconds() / 60.0
        if deltas.empty or deltas.isna().all():
            to_delete.append(row.chip_id)
            continue
        i_min = deltas.idxmin()
        cand = sub.loc[i_min]
        dist = _haversine_m(float(row.lat), float(row.lon), float(cand.get("lat", np.nan)), float(cand.get("lon", np.nan)))
        if (deltas.loc[i_min] > float(time_tol_min)) or (np.isnan(dist)) or (dist > float(dist_tol_m)):
            to_delete.append(row.chip_id)

    if not to_delete:
        logger.info("Prune by AIS: nothing to delete")
        return

    delete_set = set(to_delete)
    kept = tiles[~tiles["chip_id"].isin(delete_set)].copy()
    removed = tiles[tiles["chip_id"].isin(delete_set)].copy()

    # Delete artifacts and update daily FeatureCollections
    for row in removed.itertuples(index=False):
        tif_path = Path(row.path)
        base = tif_path.with_suffix("")
        geojson = base.with_suffix(".geojson")
        meta = base.with_suffix(".json")
        quick = tif_path.with_name(tif_path.stem + "_quicklook.jpg")
        for p in [tif_path, geojson, meta, quick]:
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {p}: {e}")
        # Update daily FC for this chip date
        try:
            date_str = pd.to_datetime(row.scene_datetime, utc=True).strftime("%Y%m%d")
            daily_fc = root / cfg["storage"]["layout"]["daily_tiles_geojson"].format(date=date_str)
            if daily_fc.exists():
                with open(daily_fc, "r", encoding="utf-8") as f:
                    fc = json.load(f)
                fc["features"] = [f for f in fc.get("features", []) if f.get("properties", {}).get("chip_id") != row.chip_id]
                with open(daily_fc, "w", encoding="utf-8") as f:
                    json.dump(fc, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed updating daily geojson for {row.chip_id}: {e}")

    kept.to_parquet(tiles_index, index=False)
    logger.info(f"Pruned {len(removed)} chips by AIS tolerances (time≤{time_tol_min} min, dist≤{dist_tol_m} m)")
