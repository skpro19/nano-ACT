"""Synthetic end-to-end action contracts; no simulator, GPU or backbone download."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from scripts import action_modes as modes
from scripts import extract_act_dataset as extract
from scripts import rollout, slim_act_dataset, train_v2
from scripts.action_checkpoint import validate_dataset_representation
from scripts.dataset import CanPhDataset
from scripts.models.act_v2 import model as act_model
from scripts.utils import analyze_z, analyze_cvae_collapse, z_sensitivity
from scripts.utils.merge_act_datasets import merge_datasets
from scripts.utils.subsample_act_dataset import subsample_dataset


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


def make_dataset(*, path: Path, commands: bool = True) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(source_metadata())
        data.attrs["source_env_args"] = json.dumps(source_metadata())
        data.attrs["supported_action_modes"] = json.dumps([modes.JOINT, modes.OSC])
        for index in range(2):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["num_samples"] = 3
            demo.create_dataset("states", data=np.zeros((3, 10)))
            joints = np.arange(21).reshape(3, 7).astype(np.float32) * .01 + index
            for group, offset in (("obs", 0), ("next_obs", .25)):
                obs = demo.create_group(group)
                obs.create_dataset("robot0_joint_pos", data=joints + offset)
                obs.create_dataset("robot0_gripper_qpos", data=np.tile([.02, -.02], (3, 1)))
                obs.create_dataset("agentview_image", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
            if commands:
                demo.create_dataset("actions", data=np.linspace(-.9, .9, 21).reshape(3, 7).astype(np.float32))


def tiny_model(*, action_dims: int = 8, use_z: bool = True) -> act_model.ACTV2:
    with patch.object(act_model, "ImageEncoder", side_effect=lambda **kwargs: nn.Identity()):
        return act_model.ACTV2(
            d_model=8, nhead=2, num_layers=1, z_dims=2,
            action_chunk_size=4, proprio_dims=8, action_dims=action_dims, use_z=use_z,
        )


@pytest.mark.parametrize("mode,dims", [(modes.JOINT, 8), (modes.OSC, 7)])
def test_dataset_alignment_normalization_and_padding(*, tmp_path, mode, dims) -> None:
    path = tmp_path / "data.hdf5"
    make_dataset(path=path)
    dataset = CanPhDataset(file=str(path), image_keys=("agentview_image",), k=4, action_mode=mode)
    assert dataset.action_dims == dims
    assert len(dataset) == 6
    sample = dataset[2]
    assert sample["proprio"].shape == (1, 8)
    assert sample["target_actions"].shape == (4, dims)
    np.testing.assert_array_equal(sample["action_mask"], [1, 0, 0, 0])
    recovered = sample["target_actions"].numpy() * dataset.normalization.action_std + dataset.normalization.action_mean
    with h5py.File(path) as file:
        expected = modes.target_actions(demo=file["data/demo_0"], action_mode=mode)
        all_actions = np.concatenate([modes.target_actions(demo=demo, action_mode=mode) for demo in file["data"].values()])
    np.testing.assert_allclose(recovered[0], expected[2], atol=1e-6)
    np.testing.assert_allclose(recovered[1:], 0, atol=1e-6)
    np.testing.assert_allclose(dataset.normalization.action_mean, all_actions.mean(axis=0), atol=1e-6)
    joint = CanPhDataset(file=str(path), image_keys=("agentview_image",))
    np.testing.assert_array_equal(dataset.normalization.proprio_mean, joint.normalization.proprio_mean)


@pytest.mark.parametrize("path,bad", [
    ("obs/robot0_joint_pos", np.zeros((2, 7))),
    ("obs/robot0_gripper_qpos", np.zeros((3, 1))),
    ("actions", np.zeros((3, 8))),
    ("actions", np.full((3, 7), np.nan)),
    ("actions", np.full((3, 7), 2.)),
    ("obs/agentview_image", np.zeros((2, 4, 4, 3))),
    ("obs/agentview_image", np.full((3, 4, 4, 3), np.inf)),
    ("obs/robot0_joint_pos", np.full((3, 7), b"bad")),
])
def test_preflight_checks_later_demos(*, tmp_path, path, bad) -> None:
    file_path = tmp_path / "data.hdf5"
    make_dataset(path=file_path)
    with h5py.File(file_path, "r+") as file:
        demo = file["data/demo_1"]
        del demo[path]
        demo.create_dataset(path, data=bad)
    with pytest.raises(ValueError, match="demo_1"):
        CanPhDataset(file=str(file_path), image_keys=("agentview_image",), action_mode=modes.OSC)


def test_joint_without_commands_and_missing_osc_extraction_guidance(*, tmp_path) -> None:
    path = tmp_path / "images_only.hdf5"
    make_dataset(path=path, commands=False)
    CanPhDataset(file=str(path), image_keys=("agentview_image",))
    with pytest.raises(ValueError, match="re-extract.*extract_act_dataset.py"):
        CanPhDataset(file=str(path), image_keys=("agentview_image",), action_mode=modes.OSC)


@pytest.mark.parametrize("key,value", [
    ("type", "OSC_POSITION"), ("control_delta", False),
    ("input_ref_frame", "base"), ("impedance_mode", "variable"),
    ("input_max", None), ("output_min", [0, 0]),
    ("input_type", "absolute"),
])
def test_rejects_unverified_osc_semantics(*, key, value) -> None:
    metadata = source_metadata()
    metadata["env_kwargs"]["controller_configs"]["body_parts"]["right"][key] = value
    with pytest.raises(ValueError):
        modes.verify_osc_metadata(env_meta=metadata)


@pytest.mark.parametrize("key", ["input_min", "output_max", "input_ref_frame", "impedance_mode", "control_delta", "gripper", "interpolation"])
def test_rejects_unknown_defaults(*, key) -> None:
    metadata = source_metadata()
    del metadata["env_kwargs"]["controller_configs"]["body_parts"]["right"][key]
    with pytest.raises(ValueError):
        modes.verify_osc_metadata(env_meta=metadata)


def test_verified_current_and_flat_legacy_metadata() -> None:
    metadata = source_metadata()
    modes.verify_osc_metadata(env_meta=metadata)
    arm = metadata["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    del arm["control_delta"]
    arm["input_type"] = "delta"
    modes.verify_osc_metadata(env_meta=metadata)
    metadata["env_kwargs"]["controller_configs"] = arm
    modes.verify_osc_metadata(env_meta=metadata)
    del metadata["env_kwargs"]["control_freq"]
    with pytest.raises(ValueError, match="control_freq"):
        modes.verify_osc_metadata(env_meta=metadata)


def test_checkpoint_legacy_defaults_and_rejects_inconsistent_contracts() -> None:
    assert modes.resolve_checkpoint_representation(checkpoint={})["mode"] == modes.JOINT
    rep = modes.representation_metadata(action_mode=modes.OSC, source_env=source_metadata())
    checkpoint = {"config": {"action_mode": modes.OSC}, "action_representation": rep}
    assert modes.resolve_checkpoint_representation(checkpoint=checkpoint) == rep
    bad = copy.deepcopy(checkpoint)
    bad["action_representation"]["action_dims"] = 8
    with pytest.raises(ValueError, match="action_dims"):
        modes.resolve_checkpoint_representation(checkpoint=bad)
    with pytest.raises(ValueError, match="disagrees"):
        modes.resolve_checkpoint_representation(checkpoint={"action_representation": rep})
    with pytest.raises(ValueError, match="missing action_representation"):
        modes.resolve_checkpoint_representation(checkpoint={"config": {"action_mode": modes.OSC}})


@pytest.mark.parametrize("key,value", [
    ("action_mean", [0] * 8), ("proprio_std", [1] * 7),
    ("action_std", [0] * 7), ("action_mean", [np.nan] * 7),
])
def test_checkpoint_normalization_validation(*, key, value) -> None:
    rep = modes.representation_metadata(action_mode=modes.OSC, source_env=source_metadata())
    stats = {"proprio_mean": [0] * 8, "proprio_std": [1] * 8, "action_mean": [0] * 7, "action_std": [1] * 7}
    stats[key] = value
    with pytest.raises(ValueError, match=key):
        rollout.normalization_from_checkpoint(checkpoint={
            "config": {"action_mode": modes.OSC}, "action_representation": rep, "normalization": stats,
        })


@pytest.mark.parametrize("use_z", [True, False])
def test_independent_dimensions_forward_backward_and_checkpoint(*, tmp_path, use_z) -> None:
    torch.set_num_threads(1)
    model = tiny_model(action_dims=7, use_z=use_z)
    proprio = torch.randn(2, 1, 8)
    actions = torch.randn(2, 4, 7)
    images = torch.randn(2, 4, 8)
    predictions, _, _ = model(proprio=proprio, actions=actions, img=images)
    assert predictions.shape == actions.shape
    predictions.square().mean().backward()
    assert model.action_head.weight.grad is not None
    model.action_representation = modes.representation_metadata(action_mode=modes.OSC, source_env=source_metadata())
    stats = {"proprio_mean": [0] * 8, "proprio_std": [1] * 8, "action_mean": [0] * 7, "action_std": [1] * 7}
    checkpoint_path = tmp_path / "checkpoint.pt"
    train_v2.save_checkpoint(
        path=checkpoint_path, run_name="test", config={"action_mode": modes.OSC, "use_z": use_z, "action_chunk_size": 4},
        dataset="synthetic", epoch=0, global_step=1, model=model,
        loss_running_avg=0., normalization=SimpleNamespace(as_checkpoint_dict=lambda: stats),
    )
    with patch.object(rollout, "ACTV2", side_effect=lambda **kwargs: tiny_model(action_dims=kwargs["action_dims"], use_z=kwargs["use_z"])):
        restored, normalization = rollout.load_model(
            checkpoint=torch.load(checkpoint_path, weights_only=False), device=torch.device("cpu"), use_z=use_z, action_chunk_size=4,
        )
    model.eval()
    torch.testing.assert_close(model.infer(proprio=proprio, img=images), restored.infer(proprio=proprio, img=images))
    assert normalization.action_mean.shape == (7,)
    with patch.object(analyze_z, "ACTV2", side_effect=lambda **kwargs: tiny_model(action_dims=kwargs["action_dims"], use_z=kwargs["use_z"])):
        analyzed, _, _ = analyze_z.load_model(checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    assert analyzed.action_dims == 7


def test_joint_state_dict_remains_compatible() -> None:
    model = tiny_model()
    assert model.action_head.out_features == 8
    other = tiny_model(action_dims=8, use_z=False)
    other.load_state_dict(model.state_dict(), strict=True)


def test_osc_execution_is_direct_bounded_input_not_scaled_pose() -> None:
    rep = modes.representation_metadata(action_mode=modes.OSC, source_env=source_metadata())
    command = np.array([.8, -.4, .3, -.2, .1, .5, -1])
    actual = rollout.model_action_to_sim(model_action=command, obs={}, action_representation=rep)
    np.testing.assert_allclose(actual, command)
    bounded = rollout.model_action_to_sim(model_action=command * 10, obs={}, action_representation=rep)
    np.testing.assert_allclose(bounded, np.clip(command * 10, -1, 1))
    config = rollout.make_rollout_controller_config(action_representation=rep)
    arm = config["body_parts"]["right"]
    assert arm["input_type"] == "delta"
    assert arm["output_max"] == [.05, .05, .05, .5, .5, .5]
    assert "input_type" not in rep["source_env_args"]["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    with pytest.raises(ValueError, match="finite"):
        rollout.model_action_to_sim(model_action=np.full(7, np.nan), obs={}, action_representation=rep)


def test_source_metadata_overrides_regenerated_env_defaults(*, tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "extracted.hdf5"
    make_dataset(path=source)
    regenerated = source_metadata()
    regenerated["env_kwargs"]["controller_configs"]["body_parts"]["right"]["input_ref_frame"] = "base"
    env = SimpleNamespace(serialize=lambda: regenerated)

    def trajectory(**kwargs):
        with h5py.File(source) as file:
            demo = file["data/demo_0"]
            observations = {name: ds[:] for name, ds in demo["obs"].items()}
            next_observations = {name: ds[:] for name, ds in demo["next_obs"].items()}
        return {"states": kwargs["states"], "obs": observations, "next_obs": next_observations, "initial_state_dict": {}}, None

    with patch.object(extract, "load_extraction_module", return_value=SimpleNamespace(extract_trajectory=trajectory)), \
            patch.object(extract.FileUtils, "get_env_metadata_from_dataset", return_value=source_metadata()), \
            patch.object(extract.EnvUtils, "create_env_for_data_processing", return_value=env), \
            patch.object(extract.EnvUtils, "is_robosuite_env", return_value=False):
        extract.extract_act_dataset(
            input_path=source, output_path=output, cameras=["agentview"], camera_height=4,
            camera_width=4, filter_key=None, num_demos=None, compress=False,
        )
    dataset = CanPhDataset(file=str(output), image_keys=("agentview_image",), action_mode=modes.OSC)
    assert dataset.action_representation["source_env_args"] == source_metadata()
    with h5py.File(source) as original, h5py.File(output) as rendered:
        np.testing.assert_array_equal(original["data/demo_1/actions"][:], rendered["data/demo_1/actions"][:])
        assert modes.OSC in json.loads(rendered["data"].attrs["supported_action_modes"])
        assert json.loads(rendered["data"].attrs["env_args"]) == regenerated


def test_slim_and_merge_preserve_commands_and_semantics(*, tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    slim = tmp_path / "slim.hdf5"
    merged = tmp_path / "merged.hdf5"
    make_dataset(path=source)
    slim_act_dataset.slim_dataset(
        input_path=source, output_path=slim, keep_next_obs_images=False,
        keep_sim_replay=False, keep_mask=False,
    )
    merge_datasets(first_path=source, second_path=slim, output_path=merged)
    dataset = CanPhDataset(file=str(merged), image_keys=("agentview_image",), action_mode=modes.OSC)
    assert len(dataset) == 12
    with h5py.File(source) as original, h5py.File(slim) as reduced:
        np.testing.assert_array_equal(original["data/demo_0/actions"][:], reduced["data/demo_0/actions"][:])
        assert original["data"].attrs["source_env_args"] == reduced["data"].attrs["source_env_args"]
    with h5py.File(slim, "r+") as file:
        changed = source_metadata()
        changed["env_kwargs"]["control_freq"] = 10
        file["data"].attrs["source_env_args"] = json.dumps(changed)
    with pytest.raises(ValueError, match="source_env_args"):
        merge_datasets(first_path=source, second_path=slim, output_path=tmp_path / "invalid.hdf5")


def test_analysis_inputs_use_commands_and_checkpoint_chunk_size(*, tmp_path) -> None:
    path = tmp_path / "data.hdf5"
    make_dataset(path=path)
    dataset = CanPhDataset(file=str(path), image_keys=("agentview_image",), k=2, action_mode=modes.OSC)
    with h5py.File(path) as file:
        proprio, actions = analyze_cvae_collapse.gather_posterior_inputs(
            hdf5=file, demo_name="demo_0", timesteps=np.array([1]),
            normalization=dataset.normalization, action_mode=modes.OSC, k=2,
        )
    assert proprio.shape == (1, 8)
    np.testing.assert_allclose(actions[0], dataset[1]["target_actions"].numpy())
    assert analyze_cvae_collapse.build_demo_lengths(dataset_path=path, k=2)["demo_0"] == 2
    validate_dataset_representation(dataset_representation=dataset.action_representation, checkpoint_representation=dataset.action_representation)
    changed = copy.deepcopy(dataset.action_representation)
    changed["source_env_args"]["env_kwargs"]["control_freq"] = 10
    with pytest.raises(ValueError, match="control_freq"):
        validate_dataset_representation(dataset_representation=changed, checkpoint_representation=dataset.action_representation)
    metrics = z_sensitivity.l1_metrics(pred=np.ones((1, 2, 7)), baseline=np.zeros((1, 2, 7)), normalization=dataset.normalization)
    assert np.isfinite(metrics["physical_gripper"])


def test_subsampling_retains_source_commands(*, tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "subset.hdf5"
    make_dataset(path=source)
    subsample_dataset(input_path=source, output_path=output, count=1, seed=4)
    dataset = CanPhDataset(file=str(output), image_keys=("agentview_image",), action_mode=modes.OSC)
    assert len(dataset) == 3
    assert dataset.action_representation["source_env_args"] == source_metadata()


def test_osc_does_not_require_achieved_next_joint_targets(*, tmp_path) -> None:
    path = tmp_path / "commands_only.hdf5"
    make_dataset(path=path)
    with h5py.File(path, "r+") as file:
        for demo in file["data"].values():
            del demo["next_obs"]
    CanPhDataset(file=str(path), image_keys=("agentview_image",), action_mode=modes.OSC)
    with pytest.raises(ValueError, match="next_obs/robot0_joint_pos"):
        CanPhDataset(file=str(path), image_keys=("agentview_image",))


@pytest.mark.parametrize("path", ["next_obs/robot0_joint_pos", "next_obs/robot0_gripper_qpos"])
def test_joint_targets_preflight_all_demos(*, tmp_path, path) -> None:
    file_path = tmp_path / "joint.hdf5"
    make_dataset(path=file_path)
    with h5py.File(file_path, "r+") as file:
        file[f"data/demo_1/{path}"][0] = np.nan
    with pytest.raises(ValueError, match="demo_1"):
        CanPhDataset(file=str(file_path), image_keys=("agentview_image",))


@pytest.mark.parametrize("mutation", ["missing_image", "wrong_length", "wrong_image_shape"])
def test_observation_preflight(*, tmp_path, mutation) -> None:
    path = tmp_path / "observations.hdf5"
    make_dataset(path=path)
    with h5py.File(path, "r+") as file:
        demo = file["data/demo_1"]
        if mutation == "wrong_length":
            demo.attrs["num_samples"] = 4
        else:
            del demo["obs/agentview_image"]
            if mutation == "wrong_image_shape":
                demo.create_dataset("obs/agentview_image", data=np.zeros((3, 5, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="demo_1"):
        CanPhDataset(file=str(path), image_keys=("agentview_image",))


@pytest.mark.parametrize("record", [False, True])
def test_osc_rollout_and_video_recording_execute_direct_commands(*, record) -> None:
    from scripts.utils import record_rollout_episodes as recording

    representation = modes.representation_metadata(action_mode=modes.OSC, source_env=source_metadata())
    model = SimpleNamespace(action_representation=representation)
    commands = np.tile([.8, -.4, .3, -.2, .1, .5, -1], (4, 1)).astype(np.float32)
    executed = []

    def step(action):
        executed.append(action)
        return {}, 0., False, {"is_success": {"task": False}}

    env = SimpleNamespace(reset=lambda: {}, step=step)
    kwargs = dict(
        model=model, env=env, device=torch.device("cpu"), normalization=None,
        image_keys=("agentview_image",), action_chunk_size=4, horizon=6,
        terminate_on_success=True, video_writer=None,
    )
    if record:
        with patch.object(recording, "predict_action_chunk", return_value=commands):
            recording.record_episode(
                **kwargs, camera_names=("agentview",), frame_skip=1, render_height=4, render_width=4,
            )
    else:
        with patch.object(rollout, "predict_action_chunk", return_value=commands):
            rollout.run_rollout(**kwargs, render=False, video_skip=1, n_action_steps=2)
    assert len(executed) == 6
    np.testing.assert_allclose(executed, np.tile(commands[0], (6, 1)))


def test_rollout_environment_uses_checkpoint_source_semantics_not_dataset_defaults(*, tmp_path) -> None:
    import robomimic.utils.env_utils as env_utils
    import robomimic.utils.file_utils as file_utils

    source = source_metadata()
    regenerated = copy.deepcopy(source)
    regenerated["env_kwargs"]["control_freq"] = 10
    regenerated["env_kwargs"]["controller_configs"]["body_parts"]["right"]["input_ref_frame"] = "base"
    representation = modes.representation_metadata(action_mode=modes.OSC, source_env=source)
    with patch.object(file_utils, "get_env_metadata_from_dataset", return_value=regenerated), \
            patch.object(env_utils, "set_env_specific_obs_processing"), \
            patch.object(rollout, "initialize_obs_modalities"):
        actual = rollout.make_rollout_env_meta(
            dataset_path=tmp_path / "rendered.hdf5", camera_names=("agentview",),
            camera_height=4, camera_width=4, action_representation=representation,
        )
    kwargs = actual["env_kwargs"]
    assert kwargs["control_freq"] == 20
    assert kwargs["controller_configs"]["body_parts"]["right"]["input_ref_frame"] == "world"
    assert kwargs["controller_configs"]["body_parts"]["right"]["kp"] == 150
    assert kwargs["camera_heights"] == 4
    assert source == source_metadata()


def test_one_cpu_training_step_logs_osc_input_metrics(*, tmp_path) -> None:
    path = tmp_path / "commands.hdf5"
    make_dataset(path=path)
    dataset = CanPhDataset(file=str(path), image_keys=("agentview_image",), k=4, action_mode=modes.OSC)
    sample = dataset[0]
    batch = {
        "proprio": sample["proprio"].unsqueeze(0),
        "target_actions": sample["target_actions"].unsqueeze(0),
        "action_mask": sample["action_mask"].unsqueeze(0),
        # Identity test backbone consumes precomputed image tokens.
        "images": torch.randn(1, 4, 8),
    }
    config = {
        "seed": 0, "batch_size": 1, "lr": 1e-4, "action_mode": modes.OSC,
        "action_chunk_size": 4, "image_keys": ["agentview_image"],
        "action_loss": "l1", "steps": 1, "beta": .01, "beta_start": 0.,
        "beta_warmup_steps": 0, "checkpoint_every": 1, "use_z": False,
        "tensorboard": {name: name in {"denorm_l1", "ranges"} for name in train_v2.TENSORBOARD_NAMESPACES},
        "rollout": {"episodes": 1, "horizon": 2, "seed": 0, "n_action_steps": [2]},
    }
    writer = MagicMock()
    model = tiny_model(action_dims=7, use_z=False)
    cpu = torch.device("cpu")
    with patch.object(train_v2.torch.cuda, "is_available", return_value=True), \
            patch.object(train_v2.torch.cuda, "get_device_name", return_value="test CPU"), \
            patch.object(train_v2.torch, "device", return_value=cpu), \
            patch.object(train_v2, "DataLoader", return_value=[batch]), \
            patch.object(train_v2, "ACTV2", return_value=model), \
            patch.object(train_v2, "RUNS_ROOT", tmp_path / "runs"), \
            patch.object(train_v2, "CHECKPOINTS_ROOT", tmp_path / "checkpoints"), \
            patch.object(train_v2, "SummaryWriter", return_value=writer), \
            patch.object(train_v2, "save_run_config"), \
            patch.object(train_v2, "maybe_save_step_checkpoint", return_value=None):
        train_v2.train(config=config, dataset=path)
    logged = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
    assert "denorm_l1/joint" not in logged
    assert "denorm_l1/gripper" not in logged
    for tag in (
        "denorm_l1/osc_input_pose", "denorm_l1/osc_translation_input",
        "denorm_l1/osc_rotation_input", "denorm_l1/gripper_command",
        "ranges/pred_min_gripper_command", "ranges/target_max_osc_input_pose",
    ):
        assert np.isfinite(logged[tag])
