# Sentinel‑2 Ship Chips — Platform Guide

End‑to‑end platform to produce “Clay‑ready” 256×256 Sentinel‑2 chips centered on vessels observed by AIS. This guide explains what the platform produces, how it works, and how to operate it in interactive (Streamlit) and scripted (CLI/scheduler) modes. It also documents data formats, configuration, and “real AIS only” operation.

Contents
- 1) What the platform produces
- 2) How it works (high level)
- 3) Requirements and installation
- 4) Configuration (config.yaml, .env)
- 5) Streamlit UI (Batch vs Continuous)
- 6) Real AIS only operation
- 7) Chip from AIS file (Excel/CSV)
- 8) Prune by AIS (cleanup)
- 9) CLI usage and scheduling
- 10) Chipping details and data model
- 11) Troubleshooting and FAQ
- 12) Roadmap and extensibility
- 13) License

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

- AIS capture: connect to aisstream.io (WebSocket) to record vessel positions and metadata around overpass windows or at periodic intervals. Snapshots are persisted to Parquet and Excel.
- Scene search: for each AIS point to process, query Earth Search STAC for Sentinel‑2 L2A scenes near the AIS time and location. Choose the best scene (closest time, cloud cover ≤ threshold).
- Chipping: convert AIS lat/lon into the scene CRS and extract a 256×256 pixel window around the point. Read bands, apply scaling to reflectance [0–1], write a compressed, tiled GeoTIFF.
- AIS + Sanity gate (optional): reject chips whose center window is likely cloud/land or insufficient water (via SCL and NDWI thresholds).
- Index: record per‑chip rows in tiles.parquet; append to daily footprints GeoJSON.

Note on predictive operation
- Demo mode uses a “backfilled” overpass schedule so scenes exist immediately for testing.
- Production predictive mode integrates ESA Acquisition Plans or TLE propagation to schedule AIS capture precisely at overpass times, then chips 2–3 days later once L2A is published.

---

3) Requirements and installation

Prerequisites
- Python 3.11+
- Internet access to Earth Search (STAC) and aisstream.io (for real AIS)
- Windows/macOS/Linux (tested on Windows 11)

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
- satellite_tracking: overpass planning (demo uses backfill for recent days)
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

5) Streamlit UI (Batch vs Continuous)

Launch the UI
```
python -m streamlit run src/ui/app.py
```

Sidebar controls

Mode
- Batch: one‑shot run; finishes and returns to idle.
- Continuous: loop runs predict→collect→chip repeatedly at an interval; suited to long‑running service use.

Include SWIR
- Toggle to include SWIR (B11,B12) resampled to 10 m appended after RGBN.

AIS source
- Batch mode: “Use synthetic AIS (dry‑run)” can be toggled for local testing. Turn OFF for real AIS.
- Continuous mode: synthetic is disabled; always real AIS.

Limits
- Max AIS ships to process: pre‑filter of how many AIS snapshots to attempt.
- Bind chips cap to ships cap (1:1): enabled by default; hides “Max chips” and sets max_chips=max_ships.
- Max chips this run: only visible when “Bind 1:1” is OFF; hard cap on produced chips.

Continuous loop interval
- Minutes between cycles (predict→collect→chip). You can stop at any time.

STAC controls
- Max cloud cover (%) filter for scene search
- Search window (hours, ± around AIS time)

AIS+Sanity gate
- Enable gate (reject cloudy/land/insufficient water center)
- Thresholds:
  - center_window_px, max_cloud_center_frac, max_land_center_frac
  - ndwi_water_thresh, min_water_center_frac

Prune by AIS
- Time tolerance (minutes) and distance tolerance (meters).
- Button to remove chips that don’t match recorded AIS within tolerances. Updates tiles.parquet and daily GeoJSON and deletes chip files.

Chip from AIS file (Excel/CSV)
- Upload a table (schema below) and click “Chip from uploaded file”. Useful for curated/recorded AIS runs or resuming after interruptions.

Main panel
- Status (Idle/Running), Logs (live tail), Quicklook gallery, Footprints map (daily), Index snapshot (tiles.parquet head).

What the UI does
- Batch Start: runs end‑to‑end with selected parameters in a background thread.
- Continuous Start: spawns a loop thread with real AIS; runs until you press Stop.
- Chip from file: runs chipping from the uploaded AIS table with the same gate/STAC settings.
- Prune by AIS: cleans mismatches with given tolerances.

---

6) Real AIS only operation

- Set AISSTREAM_API_KEY in .env.
- Batch (real AIS): disable “Use synthetic AIS (dry‑run)” and click Start.
- Continuous: real AIS enforced; start loop and stop when desired.
- For full traceability/audit, capture AIS to Excel/Parquet and then “Chip from AIS file” to ensure processing exactly matches a recorded AIS list.

Note on demo “backfill”
- The demo planner backfills recent days so scenes exist immediately for testing. In production, integrate ESA Acquisition Plans or TLE propagation to schedule AIS capture at actual overpass times and chip after a 2–3 day lag for L2A availability.

---

7) Chip from AIS file (Excel/CSV)

Upload schema
- Required columns: mmsi, lat, lon
- Optional: overpass_time (ISO8601). If missing, current UTC is used row‑wise.
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

8) Prune by AIS (cleanup)

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

9) CLI usage and scheduling

Predict overpasses (demo/backfill)
```
python -m src.cli predict-overpasses --out plan/overpass_schedule.csv
```

Collect AIS
```
# Real AIS (requires AISSTREAM_API_KEY in .env)
python -m src.cli collect-ais --schedule plan/overpass_schedule.csv --window-min 5 --config config.yaml

# Synthetic (testing):
python -m src.cli collect-ais --schedule plan/overpass_schedule.csv --window-min 5 --config config.yaml --dry-run
```

End‑to‑end (batch)
```
# Synthetic testing
python -m src.cli run --max-chips 5 --config config.yaml
```

Scheduling (recommended for production)
- Weekly: predict-overpasses
- During predicted windows: collect-ais (short run)
- After lag_days (e.g., 2–3 days): chip (use UI Continuous or a scripted batch)
- Optionally: run prune_by_ais afterward

Note: The UI Continuous mode offers a simple looping runner using real AIS. For robust production, consider OS‑level scheduling (Windows Task Scheduler, cron) calling the CLI.

---

10) Chipping details and data model

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

11) Troubleshooting and FAQ

Q: Why both “Max AIS ships to process” and “Max chips this run”?
- Ships cap is the number of AIS attempts. Some attempts may not yield chips (no scene found, gate rejected, edge, etc.). Chips cap is a hard limit on actual outputs written. To keep things simple, “Bind chips cap to ships cap (1:1)” is ON by default, and chips cap is hidden.

Q: Do I have to keep the program running for days?
- For a single batch run: no, it completes in one session.
- For predictive, real‑time collection: yes, schedule short AIS capture jobs at overpasses and chip a few days later. You can do this via OS schedulers without keeping the Streamlit UI open.

Q: Images in the UI — mockups or real?
- Real outputs from the chipper. In testing, AIS can be synthetic, but the chips are derived from actual Sentinel‑2 scenes (Earth Search). In production, disable synthetic to use only real AIS.

Common issues
- No scenes found: relax cloud cover, expand search window hours, or recheck overpass timing.
- Gate rejects all chips: adjust thresholds (max_cloud/land, min_water, NDWI).
- WebSocket drops: reconnects are handled; verify AISSTREAM_API_KEY and network.
- Raster read errors: boundless reads mitigate edges; verify coordinates and asset availability (SCL sometimes missing).

---

12) Roadmap and extensibility

- Predictive scheduler (production):
  - ESA Acquisition Plans parser; TLE propagation (sgp4/orbit-predictor installed)
- Provider abstraction:
  - Add CDSE/STAC alternative backends
- Storage:
  - S3‑compatible targets and COG writing
- Masks:
  - Persist mask rasters (cloud/water/land) alongside QA stats

---

13) License

MIT (or your chosen license).
