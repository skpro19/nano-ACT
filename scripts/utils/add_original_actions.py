"""Attach original robomimic controller actions to rendered ACT HDF5 datasets.

The ACT renderer can regenerate observations from recorded simulator states, but the
OSC action mode needs the original controller commands, which cannot be reconstructed
from observations. This script copies those commands verbatim from the official
robomimic low-dim dataset into a rendered file, producing a new "original actions"
dataset in a sibling folder.

The rendered file is copied byte-for-byte, then each demo gets an ``actions`` dataset
by matching its ``demo_<N>`` name in the low-dim source. Alignment is validated before
anything is written: simulator states and current proprioception must match exactly, and
the low-dim observations must match the rendered ``next_obs`` shifted by one step.
Provenance (source paths, sha256 digests, and the alignment checks) is recorded in
``data.attrs`` so a training run can prove which official actions it consumed.

Examples:
    # one file, written to datasets/can/mh/original_actions/
    uv run python scripts/utils/add_original_actions.py \\
        --actions datasets/can/mh/low_dim_v15.hdf5 \\
        --images datasets/can/mh/2026-10-01_03-42-46_better_agentview.hdf5

    # every rendered MH file (low_dim_v15.hdf5 and demo_v15.hdf5 are skipped because
    # they already contain actions)
    uv run python scripts/utils/add_original_actions.py \\
        --actions datasets/can/mh/low_dim_v15.hdf5 \\
        --images datasets/can/mh/*.hdf5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.action_modes import JOINT, OSC, verify_osc_metadata

PROPRIO_KEYS = ("robot0_joint_pos", "robot0_gripper_qpos")
DATA_ATTRS = ("source_env_args", "supported_action_modes", "action_dataset_provenance")
DEFAULT_OUTPUT_DIRNAME = "original_actions"

# Rendered ACT cameras are named after the observation image keys they contain.
CAMERA_SUFFIXES = ("agentview_robot0_eye_in_hand", "agentview")
TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")

# Alignment claims recorded in the provenance attribute.
NEXT_OBS_TOLERANCE = 1e-9
ALIGNMENT_NOTE = "states and current proprio exact; next_obs absolute tolerance 1e-9"
ACTIONS_NOTE = "verbatim original controller inputs; not synthesized from observations"
OFFICIAL_SOURCE_PREFIX = "robomimic/robomimic_datasets:v1.5"


def file_sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative_path(*, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def official_source_label(*, actions_path: Path) -> str:
    resolved = actions_path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT / "datasets")
    except ValueError:
        relative = Path(resolved.name)
    return f"{OFFICIAL_SOURCE_PREFIX}/{relative.as_posix()}"


def sorted_demo_names(*, dataset: h5py.File) -> list[str]:
    demos = list(dataset["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[1]))


def output_name(*, images_path: Path, split: str) -> str:
    """Map a rendered file to its ``robomimic_can_<split>_original_actions_<cams>`` name."""
    stem = TIMESTAMP_PREFIX.sub("", images_path.stem)
    for suffix in CAMERA_SUFFIXES:
        if stem.endswith(suffix):
            quality = stem[: -len(suffix)].rstrip("_")
            break
    else:
        raise ValueError(f"cannot derive quality and cameras from {images_path.name}")

    label = f"{split}_{quality}" if quality else split
    return f"robomimic_can_{label}_original_actions_{suffix}.hdf5"


def supported_action_modes(*, source_env_args: str) -> list[str]:
    supported = [JOINT]
    try:
        verify_osc_metadata(env_meta=json.loads(source_env_args))
    except ValueError:
        return supported
    supported.append(OSC)
    return supported


def verify_demo_alignment(*, images: h5py.File, actions: h5py.File, demo_name: str) -> None:
    """Fail loudly unless ``actions`` describes the same trajectory as ``images``."""
    images_demo = images[f"data/{demo_name}"]
    actions_demo = actions[f"data/{demo_name}"]
    if not np.array_equal(images_demo["states"][:], actions_demo["states"][:]):
        raise ValueError(f"demo {demo_name}: simulator states differ from the actions source")

    for key in PROPRIO_KEYS:
        if not np.array_equal(images_demo[f"obs/{key}"][:], actions_demo[f"obs/{key}"][:]):
            raise ValueError(f"demo {demo_name}: current {key} differs from the actions source")
        next_step = images_demo[f"next_obs/{key}"][:-1]
        source_step = actions_demo[f"obs/{key}"][1:]
        if not np.allclose(next_step, source_step, rtol=0, atol=NEXT_OBS_TOLERANCE):
            raise ValueError(f"demo {demo_name}: next_obs/{key} is not aligned with the actions source")


def build_provenance(
    *,
    images_path: Path,
    actions_path: Path,
    images_sha256: str,
    actions_sha256: str,
) -> dict:
    return {
        "source_actions": repo_relative_path(path=actions_path),
        "source_actions_sha256": actions_sha256,
        "source_images": repo_relative_path(path=images_path),
        "source_images_sha256": images_sha256,
        "official_source": official_source_label(actions_path=actions_path),
        "alignment": ALIGNMENT_NOTE,
        "actions": ACTIONS_NOTE,
    }


def add_original_actions(
    *,
    images_path: Path,
    actions_path: Path,
    output_path: Path,
    actions_sha256: str,
) -> None:
    if images_path.resolve() == actions_path.resolve():
        raise ValueError("actions source and rendered images must be different files")

    with h5py.File(images_path, "r") as images_file, h5py.File(actions_path, "r") as actions_file:
        demo_names = sorted_demo_names(dataset=images_file)
        if not demo_names:
            raise ValueError(f"{images_path} contains no demos")
        for demo_name in demo_names:
            verify_demo_alignment(
                images=images_file, actions=actions_file, demo_name=demo_name
            )
        source_env_args = actions_file["data"].attrs["env_args"]

    supported = supported_action_modes(source_env_args=source_env_args)
    provenance = build_provenance(
        images_path=images_path,
        actions_path=actions_path,
        images_sha256=file_sha256(path=images_path),
        actions_sha256=actions_sha256,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    shutil.copyfile(images_path, output_path)

    with h5py.File(output_path, "r+") as output_file, h5py.File(actions_path, "r") as actions_file:
        for demo_name in tqdm(demo_names, desc=f"adding actions to {output_path.name}"):
            actions_file.copy(
                f"data/{demo_name}/actions",
                output_file[f"data/{demo_name}"],
                name="actions",
            )
        output_file["data"].attrs["source_env_args"] = source_env_args
        output_file["data"].attrs["supported_action_modes"] = json.dumps(supported)
        output_file["data"].attrs["action_dataset_provenance"] = json.dumps(provenance)

    print(f"wrote {len(demo_names)} demos to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--actions",
        type=Path,
        required=True,
        help="official robomimic low-dim hdf5 that provides the original controller inputs",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="rendered ACT hdf5 file(s) to receive actions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "directory for the output hdf5 files (defaults to "
            f"'<images dir>/{DEFAULT_OUTPUT_DIRNAME}')"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions_sha256 = file_sha256(path=args.actions)
    split = args.actions.parent.name

    for images_path in args.images:
        if images_path.resolve() == args.actions.resolve():
            print(f"skipping {images_path}: it is the actions source")
            continue
        with h5py.File(images_path, "r") as images_file:
            demo_names = sorted_demo_names(dataset=images_file)
            if demo_names and "actions" in images_file[f"data/{demo_names[0]}"]:
                print(f"skipping {images_path}: demos already contain actions")
                continue

        output_dir = args.output_dir or images_path.parent / DEFAULT_OUTPUT_DIRNAME
        output_path = output_dir / output_name(images_path=images_path, split=split)
        if output_path.exists():
            print(f"skipping {images_path}: {output_path} already exists")
            continue

        add_original_actions(
            images_path=images_path,
            actions_path=args.actions,
            output_path=output_path,
            actions_sha256=actions_sha256,
        )


if __name__ == "__main__":
    main()
