"""Mix demos from several ACT datasets into one, recording where each came from."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.action_modes import JOINT, OSC
from scripts.utils.mix_act_datasets import (
    MIXING_PROVENANCE_ATTR,
    mix_datasets,
    parse_source,
)


def source_metadata() -> dict:
    return {"env_name": "PickPlaceCan", "type": 1, "env_kwargs": {"control_freq": 20}}


def make_source(
    *,
    path: Path,
    num_demos: int,
    tag: int = 0,
    source_env_args: str | None = None,
) -> None:
    """Write a tiny two-camera-free ACT file whose demos carry a recognisable ``tag``."""
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(source_metadata())
        data.attrs["source_env_args"] = source_env_args or json.dumps(source_metadata())
        data.attrs["supported_action_modes"] = json.dumps([JOINT, OSC])
        for index in range(num_demos):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["num_samples"] = 3
            demo.create_dataset("states", data=np.full((3, 2), tag + index, dtype=np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=np.zeros((3, 7), dtype=np.float32))
            obs.create_dataset("robot0_gripper_qpos", data=np.zeros((3, 2), dtype=np.float32))
            obs.create_dataset("agentview_image", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
            demo.create_dataset("actions", data=np.full((3, 7), tag + index, dtype=np.float32))


def read_provenance(*, path: Path) -> dict:
    with h5py.File(path) as file:
        return json.loads(file["data"].attrs[MIXING_PROVENANCE_ATTR])


def test_parse_source_splits_path_and_count() -> None:
    assert parse_source(spec="a/b.hdf5:45") == (Path("a/b.hdf5"), 45)


@pytest.mark.parametrize("spec", ["nopath", "a.hdf5:", ":5", "a.hdf5:x", "a.hdf5:0", "a.hdf5:-1"])
def test_parse_source_rejects_malformed(spec) -> None:
    with pytest.raises(ValueError):
        parse_source(spec=spec)


def test_mix_counts_actions_and_provenance(*, tmp_path) -> None:
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    output = tmp_path / "mixed.hdf5"
    make_source(path=first, num_demos=10, tag=0)
    make_source(path=second, num_demos=10, tag=100)

    mix_datasets(sources=[(first, 4), (second, 6)], output_path=output, seed=0)

    with h5py.File(output) as file:
        assert sorted(file["data"].keys()) == [f"demo_{index}" for index in range(10)]
        assert file["data"].attrs["total"] == 10 * 3
        assert file["data"].attrs["env_args"] == json.dumps(source_metadata())
        assert json.loads(file["data"].attrs["supported_action_modes"]) == [JOINT, OSC]
        actions = [file[f"data/demo_{index}/actions"][()].flat[0] for index in range(10)]
        assert all("actions" in file[f"data/demo_{index}"] for index in range(10))

    # The first four demos came from the first source, the rest from the second.
    assert sum(action < 100 for action in actions) == 4
    assert sum(action >= 100 for action in actions) == 6

    provenance = read_provenance(path=output)
    assert provenance["seed"] == 0
    assert [source["count"] for source in provenance["sources"]] == [4, 6]
    assert [len(source["demos"]) for source in provenance["sources"]] == [4, 6]
    assert provenance["sources"][0]["path"] == first.resolve().as_posix()


def test_mix_is_reproducible_per_seed(*, tmp_path) -> None:
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    make_source(path=first, num_demos=20, tag=0)
    make_source(path=second, num_demos=20, tag=100)
    outputs = [tmp_path / name for name in ("a.hdf5", "b.hdf5", "c.hdf5")]

    mix_datasets(sources=[(first, 5), (second, 5)], output_path=outputs[0], seed=0)
    mix_datasets(sources=[(first, 5), (second, 5)], output_path=outputs[1], seed=0)
    mix_datasets(sources=[(first, 5), (second, 5)], output_path=outputs[2], seed=1)

    assert read_provenance(path=outputs[0]) == read_provenance(path=outputs[1])
    assert read_provenance(path=outputs[0]) != read_provenance(path=outputs[2])


def test_mix_rejects_count_larger_than_source(*, tmp_path) -> None:
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    make_source(path=first, num_demos=2, tag=0)
    make_source(path=second, num_demos=2, tag=100)

    with pytest.raises(ValueError, match="cannot sample"):
        mix_datasets(sources=[(first, 5), (second, 1)], output_path=tmp_path / "mixed.hdf5", seed=0)


def test_mix_rejects_mismatched_action_metadata(*, tmp_path) -> None:
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    changed = source_metadata()
    changed["env_kwargs"]["control_freq"] = 10
    make_source(path=first, num_demos=3, tag=0)
    make_source(path=second, num_demos=3, tag=100, source_env_args=json.dumps(changed))

    with pytest.raises(ValueError, match="source_env_args"):
        mix_datasets(sources=[(first, 1), (second, 1)], output_path=tmp_path / "mixed.hdf5", seed=0)


def test_mix_requires_two_sources(*, tmp_path) -> None:
    first = tmp_path / "first.hdf5"
    make_source(path=first, num_demos=3, tag=0)

    with pytest.raises(ValueError, match="at least two sources"):
        mix_datasets(sources=[(first, 1)], output_path=tmp_path / "mixed.hdf5", seed=0)
