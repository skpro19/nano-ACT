"""Add original robomimic controller actions to rendered ACT datasets."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.action_modes import JOINT, OSC
from scripts.utils.add_original_actions import (
    REPO_ROOT,
    add_original_actions,
    official_source_label,
    output_name,
    repo_relative_path,
    supported_action_modes,
)


def source_metadata() -> dict:
    return {
        "env_name": "PickPlaceCan", "type": 1,
        "env_kwargs": {
            "robots": "Panda", "control_freq": 20,
            "controller_configs": {
                "type": "BASIC", "body_parts": {"right": {
                    "type": "OSC_POSE", "control_delta": True,
                    "input_ref_frame": "world", "impedance_mode": "fixed",
                    "input_min": -1, "input_max": 1,
                    "output_min": [-.05, -.05, -.05, -.5, -.5, -.5],
                    "output_max": [.05, .05, .05, .5, .5, .5],
                    "kp": 150, "damping_ratio": 1, "interpolation": None,
                    "gripper": {"type": "GRIP"},
                }},
            },
        },
    }


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("2026-10-01_03-42-46_better_agentview.hdf5",
         "robomimic_can_mh_better_original_actions_agentview.hdf5"),
        ("2026-10-01_03-44-12_better_agentview_robot0_eye_in_hand.hdf5",
         "robomimic_can_mh_better_original_actions_agentview_robot0_eye_in_hand.hdf5"),
        ("2026-10-01_04-02-24_better_okay_worse_agentview_robot0_eye_in_hand.hdf5",
         "robomimic_can_mh_better_okay_worse_original_actions_agentview_robot0_eye_in_hand.hdf5"),
    ],
)
def test_output_name_maps_quality_and_cameras(*, filename, expected) -> None:
    assert output_name(images_path=Path(filename), split="mh") == expected


def test_output_name_rejects_unknown_camera() -> None:
    with pytest.raises(ValueError, match="cannot derive quality and cameras"):
        output_name(images_path=Path("2026-10-01_03-42-46_better_side_view.hdf5"), split="mh")


def test_supported_action_modes_accepts_osc_and_rejects_missing_metadata() -> None:
    assert supported_action_modes(source_env_args=json.dumps(source_metadata())) == [JOINT, OSC]
    assert supported_action_modes(source_env_args=json.dumps({"env_name": "PickPlaceCan"})) == [JOINT]


def test_provenance_paths_are_relative_to_the_repository() -> None:
    path = REPO_ROOT / "datasets/can/mh/low_dim_v15.hdf5"
    assert repo_relative_path(path=path) == "datasets/can/mh/low_dim_v15.hdf5"
    assert official_source_label(actions_path=path) == (
        "robomimic/robomimic_datasets:v1.5/can/mh/low_dim_v15.hdf5"
    )


def write_actions_source(*, path: Path, T: int = 4) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(source_metadata())
        for index in range(2):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("states", data=np.zeros((T, 71)))
            demo.create_dataset("actions", data=np.full((T, 7), index, dtype=np.float64))
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=np.arange(T * 7).reshape(T, 7) + index)
            obs.create_dataset("robot0_gripper_qpos", data=np.tile([.02, -.02], (T, 1)) + index)


def write_rendered_images(*, path: Path, T: int = 4) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "rendered"})
        data.attrs["total"] = 2 * T
        for index in range(2):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["num_samples"] = T
            demo.create_dataset("states", data=np.zeros((T, 71)))
            joints = np.arange(T * 7).reshape(T, 7) + index
            gripper = np.tile([.02, -.02], (T, 1)) + index
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=joints)
            obs.create_dataset("robot0_gripper_qpos", data=gripper)
            obs.create_dataset("agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))
            next_obs = demo.create_group("next_obs")
            # next_obs[:-1] must equal the source obs shifted forward by one step.
            next_obs.create_dataset("robot0_joint_pos", data=np.vstack([joints[1:], joints[-1:]]))
            next_obs.create_dataset("robot0_gripper_qpos", data=np.vstack([gripper[1:], gripper[-1:]]))
            next_obs.create_dataset("agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))


def test_add_original_actions_copies_commands_and_records_provenance(*, tmp_path) -> None:
    actions_path = tmp_path / "low_dim_v15.hdf5"
    images_path = tmp_path / "2026-10-01_03-42-46_better_agentview.hdf5"
    output_path = tmp_path / "original_actions" / "out.hdf5"
    write_actions_source(path=actions_path)
    write_rendered_images(path=images_path)

    add_original_actions(
        images_path=images_path,
        actions_path=actions_path,
        output_path=output_path,
        actions_sha256="abc",
    )

    with h5py.File(output_path, "r") as output, h5py.File(actions_path, "r") as source:
        assert output["data"].attrs["env_args"] == json.dumps({"env_name": "rendered"})
        assert output["data"].attrs["source_env_args"] == source["data"].attrs["env_args"]
        assert json.loads(output["data"].attrs["supported_action_modes"]) == [JOINT, OSC]
        provenance = json.loads(output["data"].attrs["action_dataset_provenance"])
        assert provenance["source_actions_sha256"] == "abc"
        assert provenance["official_source"].endswith("low_dim_v15.hdf5")
        for demo_name in ("demo_0", "demo_1"):
            assert np.array_equal(
                output[f"data/{demo_name}/actions"][:], source[f"data/{demo_name}/actions"][:]
            )


def test_add_original_actions_rejects_misaligned_next_obs(*, tmp_path) -> None:
    actions_path = tmp_path / "low_dim_v15.hdf5"
    images_path = tmp_path / "2026-10-01_03-42-46_better_agentview.hdf5"
    write_actions_source(path=actions_path)
    write_rendered_images(path=images_path)
    with h5py.File(images_path, "r+") as file:
        file["data/demo_0/next_obs/robot0_joint_pos"][0, 0] = 999.0

    with pytest.raises(ValueError, match="next_obs/robot0_joint_pos is not aligned"):
        add_original_actions(
            images_path=images_path,
            actions_path=actions_path,
            output_path=tmp_path / "out.hdf5",
            actions_sha256="abc",
        )
