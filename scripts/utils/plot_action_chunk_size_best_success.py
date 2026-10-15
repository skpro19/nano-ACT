"""Plot the best eval success rate across beta for each action chunk size.

Each run directory must hold a ``config.json`` (with ``beta`` and
``action_chunk_size``) and one or more TensorBoard event files. For every run the
script takes the highest ``eval/<variant>/success_rate`` reached during
training, groups the runs by ``action_chunk_size``, and draws one line per chunk
size with ``beta`` on a log axis. This condenses a chunk-size x beta sweep into
a single figure.

Runs that vary any other setting (batch size, action mode, dataset) would be
mixed into the same beta axis, so the selection flags below pin the sweep to a
single configuration. Pass at least one of them, or explicit ``--run-name``s.

Examples:
    uv run python scripts/utils/plot_action_chunk_size_best_success.py \\
        --runs-dir runs/act_v2 \\
        --batch-size 8 --action-mode osc_pose_delta --dataset-contains train_90 \\
        --output assets/eval-curves/action_chunk_size_best_success_by_beta.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_layout import FIGURE_DPI, FIGURE_WIDTH

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "act_v2"
EVENT_GLOB = "events.out.tfevents.*"
CONFIG_FILENAME = "config.json"
SUCCESS_RATE_SUFFIX = "/success_rate"
FIGURE_PLOT_HEIGHT = 4.0
MARKERS = ("o", "s", "^", "D", "v")


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
        "--eval-variant",
        default=None,
        help="eval variant to read, e.g. n_action_steps_10 (default: best over all variants)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="keep only runs whose config has this batch size",
    )
    parser.add_argument(
        "--action-mode",
        default=None,
        help="keep only runs whose config has this action mode",
    )
    parser.add_argument(
        "--dataset-contains",
        default=None,
        help="keep only runs whose config dataset path contains this substring",
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
        help="optional figure title",
    )
    return parser.parse_args()


def discover_run_dirs(*, runs_dir: Path) -> list[Path]:
    return sorted({path.parent for path in runs_dir.rglob(EVENT_GLOB)})


def read_run_config(*, run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text()).get("config", {})


def best_success_rate(*, run_dir: Path, eval_variant: str | None) -> tuple[int, float] | None:
    """Return the highest success rate (and its step) logged anywhere in the run."""
    wanted_tag = None if eval_variant is None else f"eval/{eval_variant}{SUCCESS_RATE_SUFFIX}"
    best: tuple[int, float] | None = None
    for event_path in sorted(run_dir.glob(EVENT_GLOB)):
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if not (tag.startswith("eval/") and tag.endswith(SUCCESS_RATE_SUFFIX)):
                continue
            if wanted_tag is not None and tag != wanted_tag:
                continue
            for event in accumulator.Scalars(tag):
                if best is None or event.value > best[1]:
                    best = (event.step, float(event.value))
    return best


def config_matches(args: argparse.Namespace, *, config: dict[str, Any]) -> bool:
    if args.batch_size is not None and config.get("batch_size") != args.batch_size:
        return False
    if args.action_mode is not None and config.get("action_mode") != args.action_mode:
        return False
    if args.dataset_contains is not None and args.dataset_contains not in config.get("dataset", ""):
        return False
    return True


def collect_series(
    *, runs_dir: Path, args: argparse.Namespace) -> dict[int, dict[float, float]]:
    """Map action chunk size -> beta -> best success rate."""
    run_dirs = discover_run_dirs(runs_dir=runs_dir)
    if args.run_name is not None:
        wanted = set(args.run_name)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in wanted]

    series: dict[int, dict[float, float]] = {}
    for run_dir in run_dirs:
        config = read_run_config(run_dir=run_dir)
        if not config_matches(args, config=config):
            continue
        beta, chunk_size = config.get("beta"), config.get("action_chunk_size")
        if beta is None or chunk_size is None:
            continue
        best = best_success_rate(run_dir=run_dir, eval_variant=args.eval_variant)
        if best is None:
            print(f"skip {run_dir.name}: no eval success scalars")
            continue
        by_beta = series.setdefault(int(chunk_size), {})
        by_beta[float(beta)] = max(by_beta.get(float(beta), 0.0), best[1])
    return series


def plot_series(
    *, series: dict[int, dict[float, float]], output: Path, title: str) -> None:
    if not series:
        raise SystemExit("no runs with beta and action_chunk_size to plot")

    betas = sorted({beta for by_beta in series.values() for beta in by_beta})

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_PLOT_HEIGHT))
    for index, (marker, (chunk_size, by_beta)) in enumerate(zip(MARKERS, sorted(series.items()))):
        ordered = sorted(by_beta.items())
        axis.plot(
            [beta for beta, _ in ordered],
            [value for _, value in ordered],
            marker=marker,
            markersize=8,
            linewidth=2,
            label=f"k={chunk_size}",
        )
        above = index % 2 == 0
        for beta, value in ordered:
            axis.annotate(
                f"{value:.2f}",
                xy=(beta, value),
                xytext=(0, 8 if above else -14),
                textcoords="offset points",
                ha="center",
                va="bottom" if above else "top",
                fontsize=9,
            )

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(betas))
    axis.xaxis.set_major_formatter(FixedFormatter([f"{beta:g}" for beta in betas]))
    axis.xaxis.set_minor_locator(FixedLocator([]))
    axis.set_xlabel("beta (KL weight)")
    axis.set_ylabel("best eval success rate")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(title="action chunk size")
    if title:
        axis.set_title(title)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    series = collect_series(runs_dir=args.runs_dir.resolve(), args=args)
    plot_series(series=series, output=args.output, title=args.title)


if __name__ == "__main__":
    main()
