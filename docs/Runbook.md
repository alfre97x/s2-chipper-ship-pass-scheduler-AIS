# Production Runbook — Sentinel‑2 Ship Chips

This runbook summarizes how to operate the pipeline in production and recover safely after failures.

Overview
- Modes
  - Batch: one-shot predict → collect → chip. Good for controlled jobs and backfills.
  - Continuous: loops predict → collect → chip every N minutes; uses real AIS only.
- Inputs/Outputs
  - Inputs: plan/overpass_schedule.csv (auto-generated), AIS snapshots (ais/YYYYMMDD/*.parquet/.xlsx), Earth Search STAC.
  - Outputs: chips tree under dataset_root, indexes (tiles.parquet, daily FeatureCollections), QA report HTML, STAC catalog.

1) Configure
- Edit config.yaml
  - satellite_tracking.mode: auto | esa_plan | tle | backfill
  - stac: endpoint, cloud_cover_max, search_window_hours
  - tiling: chip_size_px, bands_core, bands_swir, resample_method
  - Phase 4 COG options:
    - tiling.write_cog: true|false
    - tiling.block_size: 256
    - tiling.overviews: [2, 4, 8, 16]
    - tiling.compression: deflate | zstd
  - logging.file: logs/run.log (rotating)
  - ops: retry/backoff, retry_batch_limit
- .env
  - AISSTREAM_API_KEY=… (required for real AIS capture)

2) Predict overpasses
Backfill demo
```
python -m src.cli predict-overpasses --out plan/overpass_schedule.csv --mode backfill
```
ESA CSV
```
python -m src.cli predict-overpasses --mode esa_plan --esa-csv plan/esa_acquisition_plan.csv
```
TLE propagation
```
python -m src.cli predict-overpasses --mode tle --out plan/overpass_schedule.csv
```

3) Collect AIS
Predictive (tile-constrained) with real AIS
```
python -m src.cli collect-ais --schedule plan/overpass_schedule.csv --window-min 5 --config config.yaml --no-dry-run --restrict-to-schedule
```
Synthetic (testing)
```
python -m src.cli collect-ais --schedule plan/overpass_schedule.csv --window-min 5 --config config.yaml --dry-run
```
Notes
- Predictive mode resolves active tiles and subscribes aisstream only to their bounding boxes.
- Each run writes parquet + Excel snapshots per tile/day under ais/YYYYMMDD/.

4) Chip (Batch)
End-to-end convenience
```
python -m src.cli run --config config.yaml --no-dry-run-ais --restrict-to-schedule --max-chips 10
```
From an AIS table (CSV/XLSX)
```
python -m src.cli run  # optional
python -m src.cli qa-report --config config.yaml
```
Or:
```
python -m streamlit run src/ui/app.py
```

5) Gate, QA, and Prune
- Gate: center-window checks (SCL cloud/land + NDWI water threshold). Tweak config.gate thresholds.
- QA: HTML report with KPIs and recent failures
```
python -m src.cli qa-report --config config.yaml --out dataset_root/QA/reports/manual_report.html
```
- Prune chips that don’t match AIS tolerances (time/distance)
  - Streamlit: “Prune by AIS” button
  - Programmatic: src.pipeline.prune_by_ais(config.yaml, time_tol_min=10, dist_tol_m=500)

6) Retries (Exponential backoff)
- On chip failure, ais_observations.status='failed', retry_count++, next_retry_at computed with backoff.
- Retry consumption:
  - During run_end_to_end a second pass executes due_ais(limit).
  - Manual retry-only pass:
```
python -m src.cli ops-retry --config config.yaml --limit 50 --include-swir
```

7) Publishing (Manifest and STAC)
- Manifest JSON (counts + samples)
```
python -m src.cli publish --config config.yaml --out-manifest dataset_root/manifests/dataset_manifest.json --sample-limit 100
```
- STAC Catalog
  - Relative HREFs:
```
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac
```
  - Absolute HREFs for hosting:
```
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac --absolute-urls --base-url https://example.com/my-dataset
```

8) Health and Maintenance
- Doctor (config/env/storage/deps)
```
python -m src.cli doctor --config config.yaml --db ops.db
```
- Vacuum DB and cleanup empty dirs
```
python -m src.cli vacuum --config config.yaml --db ops.db --rm-empty
```
- Logs: logs/run.log (rotating)

9) Scheduling Tips
- Use crontab/systemd/GitHub Actions for:
  - Weekly predict-overpasses
  - Short collect-ais jobs during predicted windows
  - Chipping runs after L2A lag (2–3 days)
  - QA report and STAC publish at end of day

10) Failure Recovery
- Safe to re-run: idempotency prevents duplicate chips (skips if .tif exists).
- If many failures:
  - Inspect ops.db (chip_jobs, ais_observations)
  - Run qa-report and ops-status
  - Execute ops-retry to process due failures
- If tiles.parquet corrupt, rebuild from artifacts by scanning chips tree (future utility; manual restore possible).

Appendix: Key Paths
- Chips: dataset_root/chips/{split}/{mmsi}/{YYYYMMDD}/
- Indexes: dataset_root/indexes/tiles.parquet
- Daily footprints: dataset_root/indexes/tiles_YYYYMMDD.geojson
- QA: dataset_root/QA/reports/{date}.html
- STAC: dataset_root/stac/catalog.json
