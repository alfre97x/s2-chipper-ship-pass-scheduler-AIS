# Sentinel‑2 Ship Chips — Platform Guide
[![CI](https://github.com/alfre97x/s2-chipper-ship-pass-scheduler-AIS/actions/workflows/ci.yml/badge.svg)](https://github.com/alfre97x/s2-chipper-ship-pass-scheduler-AIS/actions/workflows/ci.yml) • Docs: [Runbook](docs/Runbook.md) | [STAC](docs/STAC.md) | [CI](docs/CI.md)

End‑to‑end platform to produce “Clay‑ready” 256×256 Sentinel‑2 chips centered on vessels observed by AIS. This guide explains what the platform produces, how it works, and how to operate it in interactive (Streamlit) and scripted (CLI/scheduler) modes. It also documents data formats, configuration, “real AIS only” operation, and predictive (tile‑constrained) collection with ESA/TLE planners.

Contents
- 1) What the platform produces
- 2) How it works (high level)
- 3) Requirements and installation
- 4) Configuration (config.yaml, .env)
- 5) Streamlit UI (Batch vs Continuous)
- 6) Real AIS only operation
- 7) Predictive planners (auto, ESA Acquisition Plans, TLE, backfill)
- 8) Predictive (tile‑constrained) collection
- 9) Chip from AIS file (Excel/CSV)
- 10) Prune by AIS (cleanup)
- 11) CLI usage and scheduling
- 12) Chipping details and data model
- 13) Troubleshooting and FAQ
- 14) Roadmap and extensibility
- 15) License

---

1) What the platform produces

- 256×256 px GeoTIFF chips at 10 m GSD (float32 reflectance [0–1])
  - Core bands: B02 (blue), B03 (green), B04 (red), B08 (NIR)
  - Optional SWIR: B11 (swir16), B12 (swir22) resampled to 10 m and appended
- Per‑chip GeoJSON footprint (WGS84) with full metadata in properties
- Per‑chip JSON metadata sidecar mirroring the GeoJSON properties
- Quicklook JPG (RGB 2–98% stretch)
- Daily FeatureCollection GeoJSON (tiles_YYYYMMDD.geojson)
- Global tiles index (Parquet) with one row per chip
- AIS capture snapshots stored per day (Parquet + Excel)

Output layout (under dataset_root/)
- chips/{split}/{mmsi}/{YYYYMMDD}/{chip_id}.tif
- chips/{split}/{mmsi}/{YYYYMMDD}/{chip_id}.geojson
- chips/{split}/{mmsi}/{YYYYMMDD}/{chip_id}.json
- chips/{split}/{mmsi}/{YYYYMMDD}/{chip_id}_quicklook.jpg
- indexes/tiles_YYYYMMDD.geojson
- indexes/tiles.parquet

AIS capture snapshots
- ais/YYYYMMDD/ais_overpass_{tile_id}.parquet
- ais/YYYYMMDD/ais_overpass_{tile_id}.xlsx

---

2) How it works (high level)

- Predict overpasses: build plan/overpass_schedule.csv via a selectable planner mode:
  - backfill (demo), ESA plan CSV, or TLE propagation (see section 7)
- AIS capture: connect to aisstream.io (WebSocket) to record vessel positions and metadata. In predictive mode, the subscription is constrained to active Sentinel‑2 overpass tiles (by bbox) and windows.
- Scene search: for each AIS point to process, query Earth Search STAC (S2 L2A) near the AIS time and location. Choose best scene (closest time, cloud cover ≤ threshold).
- Chipping: convert AIS lat/lon into the scene CRS and extract a 256×256 px window around the point. Read bands, scale to reflectance [0–1], write a compressed, tiled GeoTIFF.
- AIS + Sanity gate (optional): reject chips whose center window is likely cloud/land or insufficient water (via SCL and NDWI thresholds).
- Index: record per‑chip rows in tiles.parquet; append to daily footprints GeoJSON.

L2A lag note
- Sentinel‑2 L2A scenes typically appear ~2–3 days after acquisition. For guaranteed overlap and fresh scenes, collect AIS at predicted pass times/tiles, then chip after the lag (Continuous mode can handle repeated runs).

---

3) Requirements and installation

Prerequisites
- Python 3.11+
- Internet access to Earth Search (STAC) and aisstream.io (for real AIS)
- Windows/macOS/Linux

Install dependencies
```
python -m pip install -r requirements.txt
```

Optional environment variables (.env)
- AISSTREAM_API_KEY=your_aisstream_api_key_here
- AWS credentials only if writing to S3/compatible storage (not required for Earth Search reads)

---

4) Configuration (config.yaml, .env)

Key sections (config.yaml)
- satellite_tracking:
  - mode: auto | esa_plan | tle | backfill
  - tiles: list of MGRS tile IDs (e.g., ["T30SUJ"])
  - overpass_window_minutes: +/- window around predicted time
  - prediction_days: horizon for TLE/backfill
  - tle_source: Celestrak URL (default included)
  - esa_csv_path: path to a locally downloaded ESA plan CSV
- ais: aisstream capture tuning (batch/reconnect)
- stac: endpoint, collection (sentinel-2-l2a), cloud_cover_max, search_window_hours
- tiling: chip_size_px, bands_core, bands_swir, resample method (bicubic)
- masks: enable SCL/NDWI usage for QA and gating
- gate: AIS+Sanity center gate (enable/disable and thresholds)
  - center_window_px: window size around chip center (default 64)
  - max_cloud_center_frac, max_land_center_frac
  - ndwi_water_thresh, min_water_center_frac
- storage: dataset_root and path templates for outputs and indexes
- splits: train/val/test fractions for deterministic chip split assignment

Environment (.env)
- AISSTREAM_API_KEY=... (needed for real AIS capture)

---

4b) Ops DB lifecycle & retries

What it tracks
- SQLite ops.db with two tables:
  - ais_observations(id, mmsi, lat, lon, tile_id, captured_time, overpass_time, source, provider, status, created_at, retry_count, last_error, last_attempt_at, next_retry_at)
    - id is deterministic: f"{mmsi}|{captured_time or overpass_time}|{tile_id or ''}"
  - chip_jobs(chip_id, mmsi, scene_id, status, error, created_at, last_attempt_at)

Lifecycle in the pipeline
1) AIS capture → DB
   - After collecting AIS (synthetic or real), rows are upserted into ais_observations with status='captured'. Existing status/created_at/retry metadata are preserved.
2) Start chipping
   - Before each chip attempt, insert/update chip_jobs with status='started' and bump last_attempt_at.
3) Success path
   - On successful chip write: chip_jobs.status='done'; ais_observations.status='chipped'.
4) Failure path + backoff
   - On exception or gate reject: chip_jobs.status='failed' with error, ais_observations.status='failed', retry_count++, and next_retry_at computed with exponential backoff:
     next_retry_at = now + min(initial_backoff_min * 2^retries, max_backoff_hours)
5) Idempotency
   - Filesystem pre-check: if the expected chip .tif already exists for the chip_id, skip the attempt and move on. This keeps re-runs safe and cheap.

Retry consumption pass
- run_end_to_end performs a second pass that queries due_ais(limit) for failed rows whose next_retry_at ≤ now and retries them (bounded by ops.retry_batch_limit). It reuses the same lifecycle and idempotency checks.

Configuration
- config.yaml:
  ops:
    max_retries: 5
    initial_backoff_min: 5
    max_backoff_hours: 24
    retry_batch_limit: 50
  alerts:
    webhook_url: ""
    threshold_fail_rate: 0.5

Notes
- DB is auto-initialized (init_db) and columns are added with best‑effort ALTER TABLE migrations.
- Alerts are stubbed in config for future wiring (e.g., send webhook on high failure rate).
- You can inspect ops.db with any SQLite browser if needed.

5) Streamlit UI (Batch vs Continuous)

Launch the UI
```
python -m streamlit run src/ui/app.py
```

Sidebar controls

Mode
- Batch: one‑shot run; finishes and returns to idle.
- Continuous: loop runs predict→collect→chip repeatedly at an interval; suited to long‑running service use (real AIS only).

Include SWIR
- Toggle to include SWIR (B11,B12) resampled to 10 m appended after RGBN.

AIS source
- Batch mode: “Use synthetic AIS (dry‑run)” can be toggled for local testing. Turn OFF for real AIS.
- Continuous mode: synthetic is disabled; always real AIS.

Limits
- Max AIS ships to process: pre‑filter of how many AIS snapshots to attempt.
- Bind chips cap to ships cap (1:1): enabled by default; hides “Max chips” and sets max_chips=max_ships.
- Max chips this run: only visible when “Bind 1:1” is OFF; hard cap on produced chips.

Predictive collection toggle
- Restrict to schedule (predictive): ON to collect AIS only within active overpass tiles and windows.
  - The app resolves active windows from plan/overpass_schedule.csv and queries STAC to obtain each tile’s bbox, then subscribes aisstream only to those bboxes for the remaining active window duration.
  - If no active windows exist at the current time, the listener falls back to a short global capture window.

Continuous loop interval
- Minutes between cycles (predict→collect→chip). Stop anytime.

STAC controls
- Max cloud cover (%) and search window (hours ±) for scene search.

AIS+Sanity gate
- Enable gate (reject cloudy/land/insufficient water center)
- Thresholds: center_window_px, max_cloud_center_frac, max_land_center_frac, ndwi_water_thresh, min_water_center_frac

Prune by AIS
- Time tolerance (minutes) and distance tolerance (meters).
- “Prune by AIS” removes chips that don’t match recorded AIS within tolerances; updates tiles.parquet and daily GeoJSON.

Chip from AIS file (Excel/CSV)
- Upload a table (schema below) and click “Chip from uploaded file”. Useful for curated/recorded AIS runs or resuming after interruptions/lag.

Ops Controls
- Retry pass limit: sets the maximum number of failed AIS rows to retry in one pass (bounded by ops.retry_batch_limit).
- Run Retry Pass: executes a single retry pass using exponential backoff and idempotency checks.
- Generate QA Report: writes an HTML report (tiles metrics + ops DB status) to storage.layout.qa_report. If alerts.threshold_fail_rate is exceeded and alerts.webhook_url is set, a webhook is posted.

Main panel
- Status (Idle/Running), Logs (live tail), Quicklook gallery, Footprints map (daily), Index snapshot (tiles.parquet head).

---

6) Real AIS only operation

- Set AISSTREAM_API_KEY in .env.
- Batch (real AIS): disable “Use synthetic AIS (dry‑run)” and (optionally) enable “Restrict to schedule (predictive)”.
- Continuous: real AIS enforced; enable predictive to constrain to active tiles and windows.
- For full traceability/audit, capture AIS to Excel/Parquet and then “Chip from AIS file”.

---

7) Predictive planners (auto, ESA, TLE, backfill)

Planner modes (config satellite_tracking.mode or CLI --mode)
- auto (production): try ESA download first; fallback to TLE; schedule written and cached/versioned
- backfill (demo): emits recent past windows (~10:30 UTC) for quick testing
- esa_plan (production): parse an ESA Acquisition Plan CSV and filter for selected tiles; write start/end UTC per tile
  - Provide file path via satellite_tracking.esa_csv_path or CLI --esa-csv
- tle (production): fetch S2A/S2B TLEs from Celestrak and propagate passes over tile centers for the next N days; write windows per tile/satellite
  - Configure prediction_days, overpass_window_minutes, tle_source

All modes write: plan/overpass_schedule.csv with columns [tile_id, satellite, start_time_utc, end_time_utc]

---

8) Predictive (tile‑constrained) collection

What it does
- Reads plan/overpass_schedule.csv for active windows (start_time_utc ≤ now ≤ end_time_utc)
- Resolves each active tile’s bbox from STAC (by s2:utm_zone, s2:latitude_band, s2:grid_square)
- Subscribes aisstream only to those bboxes during the remaining window duration
- Saves per‑tile AIS snapshots under ais/YYYYMMDD/ais_overpass_{tile_id}.parquet and .xlsx

Requirements and behavior
- Requires at least one active window at the time of capture. If none are active, the collector falls back to a short global capture window.
- In production, it’s recommended to use ESA or TLE planner modes for true predictive collection.

---

9) Chip from AIS file (Excel/CSV)

Upload schema
- Required columns: mmsi, lat, lon
- Optional: overpass_time (ISO8601). If missing, current UTC is used per row.
- Optional extras: captured_time, source, notes (not required)

Example CSV
```
mmsi,lat,lon,overpass_time
244660123,36.1251,-5.4302,2025-10-27T10:27:00Z
244660124,36.2104,-5.3566,2025-10-27T10:28:30Z
```

Behavior
- For each row, the platform searches scenes near time/location, chips 256×256 around the point, applies the AIS+Sanity gate, and writes outputs/indexes.

---

10) Prune by AIS (cleanup)

Purpose
- Keep only chips that match recorded AIS within time and position tolerances.

Tolerances
- Time tolerance (min): |scene_datetime − AIS time| ≤ threshold.
- Distance tolerance (m): haversine distance between chip center and AIS ≤ threshold.

Effects
- Deletes chip artifacts (.tif/.geojson/.json/_quicklook.jpg)
- Removes rows from tiles.parquet and features from tiles_YYYYMMDD.geojson

Run via UI: “Prune by AIS” button.
(Programmatic API: src.pipeline.prune_by_ais)

---

11) CLI usage and scheduling

Predict overpasses
```
# Backfill demo
python -m src.cli predict-overpasses --out plan/overpass_schedule.csv --mode backfill

# ESA Acquisition Plans (provide downloaded CSV path)
python -m src.cli predict-overpasses --mode esa_plan --esa-csv plan/esa_acquisition_plan.csv

# TLE propagation (S2A/S2B over selected tiles)
python -m src.cli predict-overpasses --mode tle --out plan/overpass_schedule.csv
```

Collect AIS
```
# Real AIS (requires AISSTREAM_API_KEY in .env), predictive tile restriction
python -m src.cli collect-ais --schedule plan/overpass_schedule.csv --window-min 5 --config config.yaml --no-dry-run --restrict-to-schedule

# Synthetic (testing):
python -m src.cli collect-ais --schedule plan/overpass_schedule.csv --window-min 5 --config config.yaml --dry-run
```

End‑to‑end (batch)
```
# Real predictive batch
python -m src.cli run --config config.yaml --no-dry-run-ais --restrict-to-schedule --max-chips 10

# Synthetic testing
python -m src.cli run --max-chips 5 --config config.yaml
```

Scheduling (recommended for production)
- Weekly: predict-overpasses (ESA or TLE mode)
- During predicted windows: collect-ais (short run) with --restrict-to-schedule
- After lag_days (e.g., 2–3 days): chip (use Continuous or a scheduled batch)
- Optionally: run prune_by_ais afterward

---

11b) Ops & QA CLI

QA report
```
python -m src.cli qa-report --config config.yaml --out dataset_root/QA/reports/manual_report.html
```

Ops status (DB + tiles)
```
python -m src.cli ops-status --config config.yaml --db ops.db
```

Retry-only pass (respect backoff; override limit if needed)
```
python -m src.cli ops-retry --config config.yaml --limit 50 --include-swir
```

Dataset manifest (lightweight JSON with counts and samples)
```
python -m src.cli publish --config config.yaml --out-manifest dataset_root/manifests/dataset_manifest.json --sample-limit 100
```

11c) STAC publishing and maintenance

STAC publish (relative hrefs)
```
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac
```

STAC publish (absolute hrefs for hosted catalogs)
```
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac --absolute-urls --base-url https://example.com/my-dataset
```
Notes
- Output is a STAC Catalog with a single Collection (chips) and one Item per chip.
- Asset roles include: data (chip GeoTIFF), thumbnail (quicklook JPEG), metadata (meta JSON, footprint GeoJSON).
- Validate with pystac (built-in validation via pystac; optional external tooling can be used).

Operational health
- Doctor (config/env/storage/deps check)
```
python -m src.cli doctor --config config.yaml --db ops.db
```

- Vacuum and cleanup (compacts DB; optionally removes empty directories from dataset_root)
```
python -m src.cli vacuum --config config.yaml --db ops.db --rm-empty
```

Configuration keys (Phase 4)
- tiling.write_cog: true|false  (write tiled GTiff with internal overviews)
- tiling.block_size: int        (e.g., 256)
- tiling.overviews: [2,4,8,16]  (overview pyramid factors)
- tiling.compression: deflate|zstd
- logging.file: logs/run.log
- logging.rotation: "10 MB"

12) Chipping details and data model

Bands and scaling
- Core (10 m): B02,B03,B04,B08
- SWIR (20 m → 10 m resampled): B11,B12
- Scaling: if uint16, reflectance = DN / 10000.0 → float32 in [0,1]
- Compression: deflate + tiled

Boundless reads
- If the 256×256 window goes beyond raster bounds, missing areas are zero‑filled; transform still reflects the window origin correctly.

CRS and transforms
- Raster in native UTM; per‑chip GeoTransform and EPSG recorded.
- GeoJSON footprint in WGS84 (lon/lat).

AIS+Sanity gate
- center_window_px (default 64)
- Cloud/land thresholds via SCL classes:
  - Cloud/shadow: 7,8,9,10; Land: 4,5
- NDWI near center:
  - NDWI = (B03 - B08) / (B03 + B08 + 1e-6)
  - Require water fraction > min_water_center_frac using threshold ndwi_water_thresh

Identifiers and splits
- chip_id = sha1(f"{mmsi}|{scene_datetime}|{round(lat,5)}|{round(lon,5)}")[:16]
- Deterministic split assignment (train/val/test) from chip_id hash and config fractions.

Indexes
- tiles.parquet row fields include:
  - chip_id, path, mmsi, lat, lon, overpass_time, scene_id, scene_datetime, cloud_cover, split, bands, has_swir, qa_cloud_frac, qa_water_frac, qa_land_frac.
- tiles_YYYYMMDD.geojson: per‑day FeatureCollection with footprint features.

---

13) Troubleshooting and FAQ

Q: Why both “Max AIS ships to process” and “Max chips this run”?
- Ships cap is the number of AIS attempts. Some attempts may not yield chips (no scene found, gate rejected, edge, etc.). Chips cap is a hard limit on actual outputs written. To keep things simple, “Bind chips cap to ships cap (1:1)” is ON by default, and chips cap is hidden.

Q: Do I have to keep the program running for days?
- For a single batch run: no, it completes in one session.
- For predictive, real‑time collection: yes, schedule short AIS capture jobs at overpasses and chip a few days later. You can do this via OS schedulers without keeping the Streamlit UI open.

Q: Are the images in the UI mockups or real?
- Real outputs from the chipper. In testing, AIS can be synthetic, but the chips are derived from actual Sentinel‑2 scenes. In production, disable synthetic to use only real AIS.

Q: Does predictive mode guarantee a scene?
- It strongly aligns AIS collection to S2 passes, but scenes can still be delayed or missing; clouds may reduce quality. The app reuses L2A lag and cloud thresholds to improve results.

Common issues
- No scenes found: relax cloud cover, expand search window hours, or recheck overpass timing.
- Gate rejects all chips: adjust thresholds (max_cloud/land, min_water, NDWI).
- WebSocket drops: reconnects are handled; verify AISSTREAM_API_KEY and network.
- Raster read errors: boundless reads mitigate edges; verify coordinates and asset availability (SCL sometimes missing).

---

14) Roadmap and extensibility

- Provider abstraction:
  - Add CDSE/STAC alternative backends
- Storage:
  - S3‑compatible targets and COG writing
- Masks:
  - Persist mask rasters (cloud/water/land) alongside QA stats

---

15) License

MIT.
