from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import json

import pandas as pd
from loguru import logger

from .pipeline import load_config

try:
    import pystac
except Exception as e:
    pystac = None
    logger.debug(f"pystac not available: {e}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel_or_abs(path: Path, base_dir: Path, absolute: bool, base_url: Optional[str] = None) -> str:
    if absolute and base_url:
        # Convert local path under base_dir to URL under base_url
        try:
            rel = path.relative_to(base_dir).as_posix()
            return f"{base_url.rstrip('/')}/{rel}"
        except Exception:
            return path.as_posix()
    elif absolute and not base_url:
        return path.resolve().as_posix()
    else:
        try:
            return path.relative_to(base_dir).as_posix()
        except Exception:
            return path.as_posix()


def publish_stac(
    cfg_path: str = "config.yaml",
    out_dir: Optional[str] = None,
    absolute_urls: bool = False,
    base_url: Optional[str] = None,
) -> Path:
    """
    Build a simple STAC Catalog + Collection + Items describing chips in tiles.parquet.
    Assets: chip tif, quicklook, meta json, footprint geojson (if they exist).
    If absolute_urls=True and base_url is set, uses base_url prefix for assets and catalog hrefs.
    Returns path to catalog.json.
    """
    if pystac is None:
        raise RuntimeError("pystac is required for STAC publishing. Please add 'pystac' to requirements and install.")

    cfg = load_config(cfg_path)
    root = Path(cfg["storage"]["root"]).resolve()
    tiles_path = root / cfg["storage"]["layout"]["tiles_index"]
    if not tiles_path.exists():
        raise FileNotFoundError(f"tiles index not found: {tiles_path}")

    df = pd.read_parquet(tiles_path)
    if df.empty:
        raise RuntimeError("tiles.parquet is empty; nothing to publish")

    stac_root = Path(out_dir) if out_dir else (root / "stac")
    stac_root.mkdir(parents=True, exist_ok=True)

    # Catalog and Collection
    cat = pystac.Catalog(
        id="s2-ship-chips",
        description="Sentinel-2 Ship Chips — Clay-ready 256x256 tiles",
        title="S2 Ship Chips",
    )

    coll = pystac.Collection(
        id="chips",
        description="Collection of Sentinel-2 ship-centered chips",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=pystac.TemporalExtent([[None, None]]),
        ),
        license="MIT",
        title="S2 Ship Chips Collection",
    )
    cat.add_child(coll)

    # Create items
    items: List[pystac.Item] = []
    for row in df.itertuples(index=False):
        try:
            chip_id = getattr(row, "chip_id")
            tif_path = Path(getattr(row, "path"))
            # Rebuild sibling paths (by convention from pipeline)
            ql_path = tif_path.with_name(tif_path.stem + "_quicklook.jpg")
            meta_path = tif_path.with_suffix(".json")
            geojson_path = tif_path.with_suffix(".geojson")

            # Derive geometry from geojson if exists, else create point geometry
            geometry = None
            bbox = None
            if geojson_path.exists():
                try:
                    gj = json.loads(geojson_path.read_text(encoding="utf-8"))
                    geometry = gj.get("geometry", None)
                    # compute bbox from geometry if possible
                    if geometry and geometry.get("type") == "Polygon":
                        coords = geometry.get("coordinates", [[]])[0]
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                except Exception:
                    geometry = None
                    bbox = None

            # Fallback to point
            lon = float(getattr(row, "lon"))
            lat = float(getattr(row, "lat"))
            if geometry is None:
                geometry = {"type": "Point", "coordinates": [lon, lat]}
                bbox = [lon, lat, lon, lat]

            dt = getattr(row, "scene_datetime", None)
            if pd.notna(dt):
                dt_iso = pd.to_datetime(dt, utc=True).isoformat()
            else:
                dt_iso = _now_iso()

            item = pystac.Item(
                id=str(chip_id),
                geometry=geometry,
                bbox=bbox,
                datetime=pd.to_datetime(dt_iso, utc=True).to_pydatetime(),
                properties={
                    "mmsi": str(getattr(row, "mmsi", "")),
                    "overpass_time": getattr(row, "overpass_time", None),
                    "scene_id": getattr(row, "scene_id", None),
                    "cloud_cover": getattr(row, "cloud_cover", None),
                    "split": getattr(row, "split", None),
                    "bands": getattr(row, "bands", None),
                    "has_swir": bool(getattr(row, "has_swir", False)),
                },
            )

            # Assets
            if tif_path.exists():
                item.add_asset(
                    "chip",
                    pystac.Asset(
                        href=_rel_or_abs(tif_path, root, absolute_urls, base_url),
                        media_type="image/tiff; application=geotiff",
                        roles=["data"],
                        title="Chip (GeoTIFF)",
                    ),
                )
            if ql_path.exists():
                item.add_asset(
                    "quicklook",
                    pystac.Asset(
                        href=_rel_or_abs(ql_path, root, absolute_urls, base_url),
                        media_type="image/jpeg",
                        roles=["thumbnail"],
                        title="Quicklook JPG",
                    ),
                )
            if meta_path.exists():
                item.add_asset(
                    "meta",
                    pystac.Asset(
                        href=_rel_or_abs(meta_path, root, absolute_urls, base_url),
                        media_type="application/json",
                        roles=["metadata"],
                        title="Chip metadata JSON",
                    ),
                )
            if geojson_path.exists():
                item.add_asset(
                    "footprint",
                    pystac.Asset(
                        href=_rel_or_abs(geojson_path, root, absolute_urls, base_url),
                        media_type="application/geo+json",
                        roles=["metadata"],
                        title="Footprint GeoJSON",
                    ),
                )

            coll.add_item(item)
            items.append(item)
        except Exception as e:
            logger.debug(f"Failed to create STAC item for row: {e}")
            continue

    # Save
    catalog_path = stac_root / "catalog.json"
    # Normalize hrefs; rely on provided asset hrefs (already absolute/relative via _rel_or_abs)
    cat.normalize_hrefs(stac_root.as_posix())
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    logger.info(f"STAC published at: {catalog_path}")
    return catalog_path
