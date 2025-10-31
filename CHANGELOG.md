# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.4.0] - 2025-10-31
### Added
- STAC publishing:
  - New module `src/stac_publish.py` to generate a STAC Catalog/Collection/Items from `dataset_root/indexes/tiles.parquet`.
  - New CLI `publish-stac` with `--out-dir`, `--absolute-urls`, `--base-url` options.
- COG-like chip writer:
  - Chips now written as tiled GeoTIFF with configurable block size, compression, and internal overviews.
  - Controlled via `tiling.write_cog`, `tiling.block_size`, `tiling.overviews`, `tiling.compression` in `config.yaml`.
- Observability:
  - Rotating file log sink configurable via `logging.file` and `logging.rotation`.
- Ops CLI:
  - `doctor` to validate environment/config/deps/storage.
  - `vacuum` to compact `ops.db` and optionally remove empty directories.
- Documentation:
  - Updated `README.md` with Phase 4 features, STAC usage, and CI badge.
  - New docs: `docs/Runbook.md`, `docs/STAC.md`, `docs/CI.md`.
  - New `CONTRIBUTING.md`.
- CI:
  - GitHub Actions workflow at `.github/workflows/ci.yml` running doctor, tests (if present), and optional STAC smoke.

### Changed
- `src/pipeline.py`:
  - Added tile bbox in-process cache.
  - Integrated COG-like write path and internal overviews.
  - Logging sink setup during `load_config`.
- `src/cli.py`:
  - Added `publish-stac`, `doctor`, `vacuum` commands.
  - Kept existing ops tools (`qa-report`, `ops-status`, `ops-retry`, `publish`) intact.

### Fixed
- STAC publishing compatibility by removing deprecated `make_all_assets_relative()` call.

### Notes
- Parallel chipping is reserved in config (`ops.parallel_chips`) but remains disabled (set to 1) for safety.
- For full COG validation, use `rio-cogeo validate`.

## [0.3.0] - 2025-10-30
### Added
- Ops DB lifecycle with retry/backoff.
- `qa_report.py` to generate QA HTML reports.
- CLI ops tools: `qa-report`, `ops-status`, `ops-retry`, `publish`.
- Streamlit UI Ops controls (Retry pass & QA report).

## [0.2.0] - 2025-10-28
### Added
- Predictive planners (auto/ESA/TLE/backfill).
- AIS collection (aisstream + synthetic), basic chipping pipeline.
- Streamlit UI (Batch vs Continuous), pruning by AIS.

## [0.1.0] - 2025-10-27
### Initial
- Project setup, initial chipper and indexing scaffolding.
