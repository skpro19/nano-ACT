"""Regression coverage for controller runtime boundaries and checkpoint sweeps."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scripts import action_modes as modes
from scripts import ablate, rollout, rollout_sweep, rollout_sweep_s3
from test_action_modes import source_metadata


@pytest.mark.parametrize("legacy", [False, True])
def test_osc_controller_runs_without_rendering(*, legacy) -> None:
    robosuite = pytest.importorskip("robosuite")
    metadata = source_metadata()
    if legacy:
        arm = metadata["env_kwargs"]["controller_configs"]["body_parts"]["right"]
        del arm["gripper"]
        metadata["env_kwargs"]["controller_configs"] = arm
    representation = modes.representation_metadata(action_mode=modes.OSC, source_env=metadata)
    config = rollout.make_rollout_controller_config(action_representation=representation)
    assert config["body_parts"]["right"]["gripper"] == {"type": "GRIP"}
    env = robosuite.make(
        "PickPlaceCan", robots="Panda", controller_configs=config,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, horizon=2,
    )
    try:
        env.reset()
        assert env.action_dim == 7
        obs, reward, done, info = env.step(np.zeros(7))
        assert np.isfinite(env.sim.data.qpos).all()
    finally:
        env.close()


@pytest.mark.parametrize("metadata", [None, [], {"env_kwargs": []},
    {"env_name": "PickPlaceCan", "type": 99, "env_kwargs": {}}])
def test_malformed_source_is_actionable(*, metadata) -> None:
    with pytest.raises(ValueError, match="OSC"):
        modes.verify_osc_metadata(env_meta=metadata)


@pytest.mark.parametrize("key,value", [("action_dims", 8), ("proprio_dims", 7)])
def test_checkpoint_config_dimensions_are_checked(*, key, value) -> None:
    checkpoint = {"config": {"action_mode": modes.OSC, key: value},
                  "action_representation": modes.representation_metadata(
                      action_mode=modes.OSC, source_env=source_metadata())}
    with pytest.raises(ValueError, match=key):
        modes.resolve_checkpoint_representation(checkpoint=checkpoint)


@pytest.mark.parametrize("mode", [modes.JOINT, modes.OSC])
def test_sweep_uses_checkpoint_settings_and_fresh_environment(*, mode) -> None:
    representation = modes.representation_metadata(
        action_mode=mode, source_env=source_metadata() if mode == modes.OSC else None)
    payload = {"config": {**{key: 1 for key in rollout.CHECKPOINT_CONFIG_KEYS},
                          "version": "v4", "use_z": False,
                          "image_keys": ["robot0_eye_in_hand_image"],
                          "action_chunk_size": 4, "action_mode": mode},
               "action_representation": representation}
    result = {"return": 1., "horizon": 2, "success": True, "truncated": False}
    with patch.object(rollout_sweep, "load_checkpoint", return_value=payload), \
         patch.object(rollout_sweep, "load_model", return_value=(MagicMock(), MagicMock())) as load, \
         patch.object(rollout_sweep, "make_rollout_env_meta", return_value={}) as meta, \
         patch.object(rollout_sweep, "create_rollout_env", side_effect=[MagicMock(), MagicMock()]) as create, \
         patch.object(rollout_sweep, "close_env") as close, \
         patch.object(rollout_sweep, "run_rollout", return_value=result) as run:
        for _ in range(2):
            rollout_sweep.evaluate_checkpoint(
                checkpoint=Path("epoch_1.pt"), dataset_path=Path("data.hdf5"),
                device=torch.device("cpu"), n_rollouts=1, horizon=2,
                terminate_on_success=True, on_screen=False, reset_seed_base=0,
                n_action_steps=2, progress=MagicMock())
        assert create.call_count == close.call_count == 2
        assert run.call_args_list[0].kwargs["env"] is not run.call_args_list[1].kwargs["env"]
        assert load.call_args.kwargs == {"checkpoint": payload, "device": torch.device("cpu"),
                                       "use_z": False, "action_chunk_size": 4}
        assert meta.call_args.kwargs["camera_names"] == ("robot0_eye_in_hand",)
        assert meta.call_args.kwargs["action_representation"] == representation
        assert run.call_args.kwargs["action_chunk_size"] == 4
        assert run.call_args.kwargs["n_action_steps"] == 2


def test_ablation_plan_keeps_validator_defaults(*, tmp_path) -> None:
    base = {"use_z": True}
    def validate(*, config: dict) -> dict:
        return {**config, "action_mode": modes.JOINT}
    with patch.object(ablate, "validate_config", side_effect=validate), \
         patch.object(ablate, "make_run_slug", return_value="test"):
        plan = ablate.build_plan(spec={"grid": {"use_z": [False]}},
                                 group_dir=tmp_path, base_config=base)
    assert plan[0]["effective"]["action_mode"] == modes.JOINT
    assert "action_mode" not in base
    paths = ablate.write_resolved(plan=plan, resolve_dir=tmp_path / "resolved")
    assert 'action_mode = "joint_position_aperture"' in paths[0].read_text()


def test_sweep_rejects_obsolete_architecture(*, capsys) -> None:
    with patch("sys.argv", ["rollout_sweep", "--checkpoint-dir", "unused",
                             "--model-version", "act_v1", "--n-action-steps", "1"]):
        with pytest.raises(SystemExit):
            rollout_sweep.parse_args()
    assert "scripts/train_v2.py" in capsys.readouterr().err


def test_s3_passes_execution_length_without_using_run_config(*, tmp_path) -> None:
    args = SimpleNamespace(version="act_v2", bucket="not-accessed", workspace=tmp_path,
        seed=0, n_rollouts=1, horizon=2, terminate_on_success=True, on_screen=False,
        dataset=Path("data.hdf5"), n_action_steps=3, continue_on_error=False)
    summary = {"num_rollouts": 1, "num_success": 1, "success_rate": 1.,
               "return_mean": 1., "horizon_mean": 2., "num_truncated": 0}
    with patch.object(rollout_sweep_s3, "download_json", return_value={"config": {"use_z": True}}), \
         patch.object(rollout_sweep_s3, "list_checkpoints", return_value=[(1, "epoch_1.pt")]), \
         patch.object(rollout_sweep_s3, "evaluate_checkpoint", return_value=summary) as evaluate, \
         patch.object(rollout_sweep_s3, "upload_artifact"), \
         patch.object(rollout_sweep_s3, "write_run_outputs"):
        rollout_sweep_s3.sweep_run(args=args, run_name="test", client=MagicMock(),
                                   device=torch.device("cpu"))
    assert evaluate.call_args.kwargs["n_action_steps"] == 3
    assert "use_z" not in evaluate.call_args.kwargs
    assert "image_keys" not in evaluate.call_args.kwargs
    assert "env" not in evaluate.call_args.kwargs
