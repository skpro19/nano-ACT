"""Materialize an HDF5 filter key as a standalone ACT dataset file.

Copies the demos listed under ``mask/<filter-key>`` verbatim from an already-rendered
ACT HDF5 (images, proprioception, ``states``, and per-demo attributes) into a new file
in the same directory, recomputing ``data/total``. No rendering is performed, so this
is a fast way to turn robomimic filter keys into standalone subsets -- for example the
data-size-ablation split keys ``50_percent_train`` (90 demos) and ``50_percent_valid``
(10 demos).

The output ``mask`` group keeps only filter keys whose demos all survived the copy, so
every retained key still resolves to real demos. The copied ``data`` attributes
(``env_args``, provenance, ...) are preserved so rollout evaluation keeps working.

Output names drop the ``<N>_percent`` ablation token from the filter key (a percentage
reads like a split ratio, and the size is already recorded) and append the number of
demos actually copied. The 50%-of-200 split keys therefore produce
``..._train_90.hdf5`` and ``..._valid_10.hdf5``.

Examples:
    # the 50% data-size train/valid split, next to the source file
    uv run python scripts/utils/extract_filter_key_dataset.py \\
        --input datasets/can/ph/robomimic_can_ph_original_actions_agentview_robot0_eye_in_hand.hdf5 \\
        --filter-key 50_percent_train 50_percent_valid

    # the MH operator-quality splits; the quality is already in the output stem, so
    # strip it to get ..._train_90.hdf5 rather than ..._better_train_90.hdf5
    uv run python scripts/utils/extract_filter_key_dataset.py \\
        --input datasets/can/mh/original_actions/robomimic_can_mh_better_original_actions_agentview_robot0_eye_in_hand.hdf5 \\
        --filter-key better_train better_valid better_operator_1_train better_operator_1_valid \\
        --strip-prefix better_
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ``data`` attributes that are recomputed per subset rather than copied from the source.
RECOMPUTED_DATA_ATTRS = ("total",)


def sorted_demo_names(*, dataset: h5py.File) -> list[str]:
    demos = list(dataset["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[1]))


def filtered_demo_names(*, dataset: h5py.File, filter_key: str) -> list[str]:
    """Return the demos referenced by ``mask/<filter_key>``, in sorted order."""
    mask_path = f"mask/{filter_key}"
    if mask_path not in dataset:
        available = sorted(dataset["mask"].keys()) if "mask" in dataset else []
        raise KeyError(f"filter key {mask_path!r} not found (available: {available})")

    tagged = {name.decode("utf-8") for name in dataset[mask_path][:]}
    return [name for name in sorted_demo_names(dataset=dataset) if name in tagged]


def output_label(*, filter_key: str, num_demos: int, strip_prefix: str = "") -> str:
    """Build the output filename label from a filter key and its demo count.

    The ``<N>_percent`` ablation token is dropped: a percentage reads like a split
    ratio and is redundant once the episode count is known, so ``50_percent_train``
    becomes ``train_90``. A key that is only a percentage (``50_percent``) becomes
    just the count.

    ``strip_prefix`` removes a leading token that is already encoded elsewhere in the
    output name (for example the operator quality in ``better_train``), so
    ``better_operator_1_train`` with ``strip_prefix="better_"`` becomes
    ``operator_1_train_45``.
    """
    base = filter_key[len(strip_prefix):] if strip_prefix and filter_key.startswith(strip_prefix) else filter_key
    base = re.sub(r"\d+_percent", "", base).strip("_")
    return f"{base}_{num_demos}" if base else str(num_demos)


def default_output_path(*, input_path: Path, label: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{label}{input_path.suffix}")


def copy_data_attrs(*, source: h5py.File, output: h5py.File, total: int) -> None:
    for attr_key, value in source["data"].attrs.items():
        if attr_key not in RECOMPUTED_DATA_ATTRS:
            output["data"].attrs[attr_key] = value
    output["data"].attrs["total"] = total


def copy_filter_keys(*, source: h5py.File, output: h5py.File, demo_names: set[str]) -> None:
    """Copy source ``mask`` keys whose demos all survived the extraction."""
    if "mask" not in source:
        return

    output_mask = output.create_group("mask")
    for key in source["mask"]:
        tagged = [name.decode("utf-8") for name in source[f"mask/{key}"][:]]
        if set(tagged) - demo_names:
            continue
        output_mask.create_dataset(key, data=np.array(sorted(tagged), dtype="S"))


def extract_filter_key_dataset(*, input_path: Path, output_path: Path, filter_key: str) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(f"output path must differ from the input path: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with h5py.File(input_path, "r") as source_file, h5py.File(output_path, "w") as output_file:
        for attr_key, value in source_file.attrs.items():
            output_file.attrs[attr_key] = value

        source_count = len(source_file["data"].keys())
        demo_names = filtered_demo_names(dataset=source_file, filter_key=filter_key)
        if not demo_names:
            raise ValueError(f"filter key {filter_key!r} selected no demos")

        data_group = output_file.create_group("data")
        total = 0
        for demo_name in tqdm(demo_names, desc=f"copying {filter_key} demos"):
            source_file.copy(f"data/{demo_name}", data_group, name=demo_name)
            total += int(source_file[f"data/{demo_name}"].attrs["num_samples"])

        copy_data_attrs(source=source_file, output=output_file, total=total)
        copy_filter_keys(source=source_file, output=output_file, demo_names=set(demo_names))

    print(
        f"wrote {len(demo_names)} of {source_count} demos ({filter_key}, {total} samples) "
        f"to {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to an already-rendered ACT hdf5 with a mask group",
    )
    parser.add_argument(
        "--filter-key",
        type=str,
        nargs="+",
        required=True,
        help="mask/<filter-key> to materialize as its own hdf5 (one output file per key)",
    )
    parser.add_argument(
        "--strip-prefix",
        type=str,
        default="",
        help=(
            "leading token removed from each filter key before the demo count is appended; "
            "use this when that token is already encoded in the output name (e.g. better_)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the output hdf5 files (defaults to the input directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent
    with h5py.File(args.input, "r") as source_file:
        for filter_key in args.filter_key:
            num_demos = len(filtered_demo_names(dataset=source_file, filter_key=filter_key))
            label = output_label(
                filter_key=filter_key, num_demos=num_demos, strip_prefix=args.strip_prefix
            )
            output_path = default_output_path(input_path=args.input, label=label)
            if output_dir != args.input.parent:
                output_path = output_dir / output_path.name
            extract_filter_key_dataset(
                input_path=args.input, output_path=output_path, filter_key=filter_key
            )


if __name__ == "__main__":
    main()
