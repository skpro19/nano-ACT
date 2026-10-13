"""Create a mixed-quality ACT dataset by sampling demos from several sources.

The multi-human (MH) quality tiers partition the CAN demonstrations into ``better``,
``okay``, and ``worse`` groups. A *mixed* dataset combines two tiers so a run can study
how a policy behaves on heterogeneous quality at a controlled size: this utility draws a
fixed number of demos from each source, merges them into one HDF5, and records exactly
which demos came from where.

Demo groups are copied verbatim (images, proprioception, ``states``, and ``actions``), so
the output keeps the schema of its sources. Demos are re-indexed ``demo_0..demo_N`` to
avoid the ``demo_0`` collisions between sources, and the ``mask`` group is dropped because
its keys reference per-source demo names and would be stale. Action semantics
(``env_args``, ``source_env_args``, ``supported_action_modes``) must agree across sources;
they are copied to the output along with a ``mixing_provenance`` attribute describing the
sources, counts, and seed.

Examples:
    uv run python scripts/utils/mix_act_datasets.py \\
        --source datasets/can/mh/original_actions/better_agentview_robot0_eye_in_hand_train_90.hdf5:45 \\
        --source datasets/can/mh/original_actions/okay_agentview_robot0_eye_in_hand_train_90.hdf5:45 \\
        --seed 42 \\
        --output datasets/can/mh/original_actions/better_okay_agentview_robot0_eye_in_hand_train_90.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.add_original_actions import file_sha256, repo_relative_path
from scripts.utils.merge_act_datasets import (
    ACTION_METADATA_KEYS,
    assert_compatible_action_metadata,
    copy_demo,
)
from scripts.utils.subsample_act_dataset import sampled_demo_names

DEFAULT_SEED = 42
MIXING_PROVENANCE_ATTR = "mixing_provenance"


def parse_source(*, spec: str) -> tuple[Path, int]:
    """Parse a ``PATH:COUNT`` source specification into a path and a positive count."""
    path_text, separator, count_text = spec.rpartition(":")
    if not separator or not path_text or not count_text.isdigit():
        raise ValueError(f"source must be 'PATH:COUNT', got {spec!r}")
    count = int(count_text)
    if count <= 0:
        raise ValueError(f"source count must be > 0, got {count} in {spec!r}")
    return Path(path_text), count


def build_mixing_provenance(
    *, sources: list[tuple[Path, int]], selection: list[list[str]], seed: int,) -> dict:
    """Describe which demos were taken from each source so the mix is reproducible."""
    return {
        "seed": seed,
        "sources": [
            {
                "path": repo_relative_path(path=path),
                "sha256": file_sha256(path=path),
                "count": count,
                "demos": demos,
            }
            for (path, count), demos in zip(sources, selection)
        ],
    }


def mix_datasets(*, sources: list[tuple[Path, int]], output_path: Path, seed: int) -> None:
    if len(sources) < 2:
        raise ValueError("mixing needs at least two sources")
    for path, _ in sources:
        if path.resolve() == output_path.resolve():
            raise ValueError(f"output path must differ from every source: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_samples = 0
    with h5py.File(output_path, "w") as output_file:
        data_group = output_file.create_group("data")
        source_files = [h5py.File(path, "r") for path, _ in sources]
        try:
            env_args = assert_compatible_action_metadata(files=source_files)
            selection = [
                sampled_demo_names(dataset=source_file, count=count, seed=seed)
                for source_file, (_, count) in zip(source_files, sources)
            ]

            index = 0
            for source_file, demos in zip(source_files, selection):
                for demo_name in tqdm(demos, desc=f"copying {Path(source_file.filename).name}"):
                    total_samples += copy_demo(
                        source_demo=source_file[f"data/{demo_name}"],
                        output_group=data_group,
                        demo_name=f"demo_{index}",
                    )
                    index += 1

            data_group.attrs["total"] = total_samples
            data_group.attrs["env_args"] = env_args
            for key in ACTION_METADATA_KEYS:
                value = source_files[0]["data"].attrs.get(key)
                if value is not None:
                    data_group.attrs[key] = value
            data_group.attrs[MIXING_PROVENANCE_ATTR] = json.dumps(
                build_mixing_provenance(sources=sources, selection=selection, seed=seed)
            )
        finally:
            for source_file in source_files:
                source_file.close()

    print(
        f"mixed {sum(count for _, count in sources)} demos from {len(sources)} sources "
        f"({total_samples} samples, seed {seed}) to {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        action="append",
        required=True,
        help="PATH:COUNT of demos to draw from a source; repeat for each source",
    )
    parser.add_argument("--output", type=Path, required=True, help="path for the mixed hdf5")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed for the (reproducible) demo selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [parse_source(spec=spec) for spec in args.source]
    mix_datasets(sources=sources, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
