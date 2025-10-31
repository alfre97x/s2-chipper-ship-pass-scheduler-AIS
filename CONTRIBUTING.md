# Contributing to Sentinel‑2 Ship Chips

Thanks for your interest in contributing!

Prerequisites
- Python 3.11+
- GDAL/rasterio system libs (see CI for Ubuntu apt packages)
- An environment with required Python packages:
```
python -m pip install -r requirements.txt
```

Project layout
- src/: Python packages and modules
- src/ui/app.py: Streamlit UI
- src/cli.py: Typer CLI entrypoints
- src/pipeline.py: Core pipeline (predict, collect AIS, chip)
- src/stac_publish.py: STAC writer
- docs/: documentation
- dataset_root/: local outputs (ignored by git)
- ais/: AIS snapshots (ignored by git)
- logs/: log files (ignored by git)

Dev workflow
1) Run environment checks
```
python -m src.cli doctor --config config.yaml --db ops.db
```

2) Run the app/CLI locally
- Streamlit UI
```
python -m streamlit run src/ui/app.py
```
- End-to-end (batch)
```
python -m src.cli run --config config.yaml --dry-run-ais --max-chips 3
```
- QA report
```
python -m src.cli qa-report --config config.yaml
```

3) Tests
- If tests are present, run:
```
pytest -q
```

Coding guidelines
- Keep functions small and composable; handle errors defensively (non-fatal where reasonable).
- Avoid hard-coded paths; read from config.yaml or environment.
- Logging: use loguru; rely on config.logging for file sink.
- Idempotency: when creating chips, check for existence first.

Pull requests
- Include a brief description and screenshots when UI is involved.
- Update docs when adding or changing CLI commands, config fields, or behavior.
- Ensure `python -m src.cli doctor` and (if applicable) `pytest` pass locally.

Commit messages
- Use concise, descriptive messages, e.g.:
  - pipeline: add COG-like writer with internal overviews
  - cli: add publish-stac/doctor/vacuum commands
  - docs: add Runbook, STAC, CI; update README

Release process
- Update CHANGELOG.md.
- Tag semver version (e.g., v0.4.0) and push tag.
- CI should be green before creating a GitHub release.
