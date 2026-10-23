"""Plot training-time eval curves for a set of runs side by side.

Reads the ``eval/<variant>/success_rate`` and ``eval/<variant>/horizon_mean``
scalars that ``scripts/train_v2.py`` writes to TensorBoard and renders them as a
row of two panels so an ablation (for example batch size) can be compared at a
glance. The canvas and panel sizes match ``plot_training_run.py`` so every curve
in the README lines up.

``--sweep-key`` picks the setting named in the legend, which also orders the
runs: ``auto`` describes the batch size (plus the camera count when the run
config records it), while ``action-chunk-size`` labels each run with its action
chunk size ``k``. Values are read from ``config.json`` and fall back to the
``bs<N>`` / ``k<N>`` token in the run directory name.

Examples:
    uv run python scripts/utils/plot_eval_curves.py \\
        --runs-dir runs/act_v2/v4 \\
        --run-name 20260929-113816_bs8_lr1e-04_... \\
        --run-name 20260929-162138_bs16_lr1e-04_... \\
        --output assets/eval-curves/batch_size_eval_curves_legend_right.png

    uv run python scripts/utils/plot_eval_curves.py \\
        --runs-dir runs/act_v2 \\
        --run-name 2026-10-22_19-19-10_osc_pose_delta_k100_bs8_... \\
        --run-name 2026-10-22_19-15-47_osc_pose_delta_k50_bs8_... \\
        --sweep-key action-chunk-size \\
        --title "action chunk size ablation" \\
        --output assets/eval-curves/action_chunk_size_eval_curves_legend_right.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_layout import FIGURE_DPI, FIGURE_HEIGHT, FIGURE_WIDTH, place_panel_row

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "act_v2"
SUCCESS_RATE_TAG = "eval/success_rate"
HORIZON_TAG = "eval/horizon_mean"
EVENT_GLOB = "events.out.tfevents.*"
CONFIG_FILENAME = "config.json"
BATCH_SIZE_PATTERN = re.compile(r"_bs(\d+)_")
CHUNK_SIZE_PATTERN = re.compile(r"_k(\d+)_")
SWEEP_KEYS = ("auto", "action-chunk-size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="directory searched recursively for run directories with event files",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="restrict to these run directory names (repeatable)",
    )
    parser.add_argument(
        "--sweep-key",
        choices=SWEEP_KEYS,
        default="auto",
        help="setting named in the legend, which also orders the runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path of the PNG figure to write",
    )
    parser.add_argument(
        "--title",
        default="",
        help="optional figure suptitle",
    )
    return parser.parse_args()


def discover_run_dirs(*, runs_dir: Path) -> list[Path]:
    run_dirs = {path.parent for path in runs_dir.rglob(EVENT_GLOB)}
    return sorted(run_dirs)


def read_scalar_series(*, run_dir: Path, tag: str) -> list[tuple[int, float]]:
    series: dict[int, float] = {}
    for event_path in sorted(run_dir.glob(EVENT_GLOB)):
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            series[event.step] = float(event.value)
    return sorted(series.items())


def read_run_config(*, run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text())
    return payload.get("config", {})


def read_int_setting(
    *, run_dir: Path, config: dict[str, Any], key: str, pattern: re.Pattern[str]) -> int | None:
    value = config.get(key)
    if value is not None:
        return int(value)
    match = pattern.search(run_dir.name)
    return int(match.group(1)) if match else None


def sweep_value(*, run_dir: Path, config: dict[str, Any], sweep_key: str) -> int | None:
    """Value of the swept setting, used both as the legend text and for ordering."""
    if sweep_key == "action-chunk-size":
        return read_int_setting(
            run_dir=run_dir, config=config, key="action_chunk_size", pattern=CHUNK_SIZE_PATTERN
        )
    return read_int_setting(
        run_dir=run_dir, config=config, key="batch_size", pattern=BATCH_SIZE_PATTERN
    )


def run_label(*, run_dir: Path, config: dict[str, Any], sweep_key: str) -> str:
    value = sweep_value(run_dir=run_dir, config=config, sweep_key=sweep_key)
    if value is None:
        return run_dir.name
    if sweep_key == "action-chunk-size":
        return f"k={value}"
    num_cameras = len(config.get("image_keys", []))
    if not num_cameras:
        return f"bs={value}"
    camera_word = "camera" if num_cameras == 1 else "cameras"
    return f"bs={value} · {num_cameras} {camera_word}"


def legend_label(*, label: str, eval_variant: str, sweep_key: str, swept_value: int | None) -> str:
    """Add the eval variant, unless it only restates the swept value (``k=100``)."""
    variant_name = eval_variant.rsplit("/", 1)[-1]
    if sweep_key == "action-chunk-size" and variant_name == f"n_action_steps_{swept_value}":
        return label
    return f"{label} · {eval_variant}"


def build_runs(*, runs_dir: Path, run_names: list[str] | None, sweep_key: str) -> list[dict[str, Any]]:
    run_dirs = discover_run_dirs(runs_dir=runs_dir)
    if run_names is not None:
        wanted = set(run_names)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in wanted]
    if not run_dirs:
        raise SystemExit(f"no run directories with event files under {runs_dir}")

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        prefixes = set()
        for event_path in sorted(run_dir.glob(EVENT_GLOB)):
            accumulator = EventAccumulator(str(event_path))
            accumulator.Reload()
            prefixes.update(
                tag.rsplit("/", 1)[0] for tag in accumulator.Tags().get("scalars", [])
                if tag.startswith("eval/") and tag.endswith("/success_rate"))
        if not prefixes:
            print(f"skip {run_dir.name}: no eval scalars")
            continue
        config = read_run_config(run_dir=run_dir)
        integer_value = sweep_value(run_dir=run_dir, config=config, sweep_key=sweep_key)
        label = run_label(run_dir=run_dir, config=config, sweep_key=sweep_key)
        for prefix in sorted(prefixes):
            runs.append({
                "label": legend_label(
                    label=label, eval_variant=prefix, sweep_key=sweep_key, swept_value=integer_value
                ),
                "swept_value": integer_value or 0,
                "success_rate": read_scalar_series(run_dir=run_dir, tag=f"{prefix}/success_rate"),
                "horizon": read_scalar_series(run_dir=run_dir, tag=f"{prefix}/horizon_mean"),
            })
    runs.sort(key=lambda run: run["swept_value"])
    return runs


def plot_runs(*, runs: list[dict[str, Any]], output: Path, title: str) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True, squeeze=False
    )
    place_panel_row(axes=list(axes[0, :]))
    success_ax, horizon_ax = axes[0, :]

    for run in runs:
        label = run["label"]
        if run["success_rate"]:
            steps, values = zip(*run["success_rate"])
            success_ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=label)
        if run["horizon"]:
            steps, values = zip(*run["horizon"])
            horizon_ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=label)

    success_ax.set_ylabel("success rate")
    success_ax.set_ylim(0.0, 1.0)
    horizon_ax.set_ylabel("mean horizon")

    for ax in (success_ax, horizon_ax):
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)

    handles, labels = success_ax.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="center right", bbox_to_anchor=(0.995, 0.5))

    if title:
        figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    runs = build_runs(
        runs_dir=args.runs_dir.resolve(), run_names=args.run_name, sweep_key=args.sweep_key
    )
    plot_runs(runs=runs, output=args.output, title=args.title)


if __name__ == "__main__":
    main()
