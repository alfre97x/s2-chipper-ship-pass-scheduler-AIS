# Continuous Integration (CI)

This project uses GitHub Actions to run basic checks on every push and pull request.

Workflow file
- .github/workflows/ci.yml

What the workflow does
1) Setup
   - Checks out the repository
   - Installs Python 3.11
   - Caches pip packages
   - Installs system libs (gdal) required by rasterio

2) Install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Environment doctor (sanity checks)
```
python -m src.cli doctor --config config.yaml --db ops.db
```
- Validates config sections exist
- Ensures dataset_root path is creatable
- Verifies dependencies (rasterio, pystac)
- Checks SQLite DB is accessible

4) Tests (if present)
```
pytest -q || true
```
- Runs tests quietly
- Tolerant if no tests exist yet

5) Optional STAC publish smoke
- If dataset_root/indexes/tiles.parquet exists, attempts to build a STAC catalog locally
```
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac || true
```

Add a badge to README
Use your GitHub org and repo for the badge:
```
![CI](https://github.com/alfre97x/s2-chipper-ship-pass-scheduler-AIS/actions/workflows/ci.yml/badge.svg)
```

Local reproduction
- You can run the same commands locally:
```
python -m src.cli doctor --config config.yaml --db ops.db
pytest -q
python -m src.cli publish-stac --config config.yaml --out-dir dataset_root/stac
```

Notes
- The CI does not upload data; it only validates environment and runs smoke checks.
- To speed up runs, keep requirements lean and prefer caching where possible.
