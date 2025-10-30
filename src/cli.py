import sys
from pathlib import Path
import typer
from loguru import logger

from .pipeline import (
    load_config,
    predict_overpasses,
    collect_ais_for_schedule,
    run_end_to_end,
)

app = typer.Typer(help="Sentinel-2 Ship Chips pipeline CLI")


@app.command("predict-overpasses")
def cli_predict_overpasses(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml"),
    out_csv: str = typer.Option("plan/overpass_schedule.csv", "--out", "-o", help="Output schedule CSV path"),
):
    """Produce an overpass schedule CSV (MVP heuristic; upgrade to ESA/TLE later)."""
    cfg = load_config(config)
    out = Path(out_csv)
    predict_overpasses(cfg, out)
    logger.info(f"Overpass schedule written: {out}")


@app.command("collect-ais")
def cli_collect_ais(
    config: str = typer.Option("config.yaml", "--config", "-c"),
    schedule_csv: str = typer.Option("plan/overpass_schedule.csv", "--schedule", "-s"),
    window_min: int = typer.Option(5, "--window-min", "-w"),
    out_dir: str = typer.Option(None, "--out-dir"),
    dry_run: bool = typer.Option(True, help="Generate synthetic AIS if no API key or for testing"),
    restrict_to_schedule: bool = typer.Option(False, "--restrict-to-schedule/--no-restrict-to-schedule", help="Subscribe aisstream only to active overpass tile bounding boxes"),
):
    """Collect AIS around scheduled overpasses (aisstream or synthetic)."""
    cfg = load_config(config)
    out = collect_ais_for_schedule(
        cfg,
        Path(schedule_csv),
        window_min=window_min,
        out_dir=Path(out_dir) if out_dir else None,
        dry_run=dry_run,
        restrict_to_schedule=restrict_to_schedule,
    )
    logger.info(f"AIS parquet: {out}")


@app.command("run")
def cli_run(
    config: str = typer.Option("config.yaml", "--config", "-c"),
    dry_run_ais: bool = typer.Option(True, help="Use synthetic AIS for testing"),
    max_chips: int = typer.Option(5, help="Maximum chips to generate in one run"),
    restrict_to_schedule: bool = typer.Option(False, "--restrict-to-schedule/--no-restrict-to-schedule", help="Collect AIS only within active overpass tile bounding boxes"),
):
    """End-to-end: predict -> collect AIS -> STAC search -> chip tiles."""
    run_end_to_end(cfg_path=config, dry_run_ais=dry_run_ais, max_chips=max_chips, restrict_to_schedule=restrict_to_schedule)


if __name__ == "__main__":
    app()
