# STAC Catalog — Sentinel‑2 Ship Chips

This document describes how the project publishes a STAC Catalog, the structure of the catalog, validation, and hosting options.

Overview
- Catalog layout is built from the global tiles index (dataset_root/indexes/tiles.parquet).
- One STAC Catalog containing a single Collection (chips) and one Item per chip.
- Assets on each Item:
  - chip: GeoTIFF (tiled, compressed; COG-like settings)
  - quicklook: JPEG preview
  - meta: JSON sidecar (per-chip metadata)
  - footprint: GeoJSON feature (footprint polygon with properties)

Publishing commands
- Relative HREFs (local browsing under dataset_root/stac)
```
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac
```

- Absolute HREFs (hosted catalog with public URLs)
```
python -m src.cli publish-stac --config config.yaml \
  --out-dir dataset_root/stac \
  --absolute-urls \
  --base-url https://example.com/my-dataset
```

Notes
- When absolute URLs are used, asset hrefs are normalized against the provided base URL; ensure the chip/quicklook/meta/geojson files are hosted at the same relative paths under that base_url.
- The CLI does not upload files; perform separate hosting/sync (e.g., S3 sync, GitHub Pages publish, or static web server).

Catalog structure (example)
- dataset_root/stac/catalog.json
- dataset_root/stac/chips/collection.json
- dataset_root/stac/chips/<item_id>.json (one per chip)

Item geometry and properties
- geometry: Footprint polygon from the per-chip GeoJSON when available; otherwise a Point geometry at the chip center (lon,lat).
- bbox: Derived from geometry.
- datetime: scene_datetime from tiles.parquet; fallback to now in UTC if missing.
- properties include:
  - mmsi, overpass_time, scene_id, cloud_cover
  - split, bands, has_swir
  - optional QA metrics are available in the GeoJSON and in indexes (cloud/water/land fractions)

Validation
- Validate with pystac’s built-in validators:
```
python -c "import pystac; c=pystac.Catalog.from_file('dataset_root/stac/catalog.json'); c.validate_all()"
```
- Optional external tools:
  - stac-validator (PyPI)
  - stac-check

Integration with COGs
- Chips are written as tiled GeoTIFF with internal overviews when config.tiling.write_cog=true.
- Media type used in STAC is "image/tiff; application=geotiff".
- For full COG conformance, you may validate using rio-cogeo:
```
rio cogeo validate dataset_root/chips/<...>.tif
```

Hosting options
- S3-compatible object storage:
  - Upload dataset_root contents; set --absolute-urls with a CDN or website endpoint base URL.
- GitHub Pages / static hosting:
  - Publish the stac directory as a static site; ensure absolute URLs for assets or host assets alongside the catalog.
- Local HTTP server:
  - `python -m http.server` from dataset_root (or from a higher-level directory) and use `--base-url http://localhost:8000/stac`.

Troubleshooting
- Missing assets in Items:
  - Ensure quicklook, meta, and geojson files exist next to the .tif files.
- Empty catalog:
  - Check dataset_root/indexes/tiles.parquet exists and contains rows.
- Validation errors:
  - Confirm Item fields match types and coordinate order is [lon, lat].
  - Ensure CRS and footprint geometry are valid polygons.
