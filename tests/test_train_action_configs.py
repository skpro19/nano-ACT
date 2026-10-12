"""Ensure action-specific base configs preserve shared training defaults."""
from pathlib import Path

from scripts.ablate import build_plan, load_spec
from scripts.action_modes import JOINT, OSC, action_dims
from scripts.train_v2 import load_config, read_config_file


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/train/act_v2"
OSC_DATASET = (
    "datasets/can/ph/"
    "robomimic_can_ph_original_actions_agentview_robot0_eye_in_hand.hdf5"
)


def test_osc_base_changes_only_action_mode_and_dataset() -> None:
    baseline = read_config_file(path=CONFIG_DIR / "BS-32/BASE.toml")
    osc = read_config_file(path=CONFIG_DIR / "OSC-POSE/BASE.toml")
    assert osc == {**baseline, "action_mode": OSC, "dataset": OSC_DATASET}


def test_osc_base_is_a_valid_training_config() -> None:
    config = load_config(path=CONFIG_DIR / "OSC-POSE/BASE.toml")
    assert config["action_mode"] == OSC
    assert config["dataset"] == OSC_DATASET
    assert action_dims(action_mode=config["action_mode"]) == 7
    assert config["image_keys"] == ["agentview_image", "robot0_eye_in_hand_image"]
    assert config["use_z"] is False
    assert config["batch_size"] == 32
    assert config["rollout"]["n_action_steps"] == [10]


def test_action_mode_and_chunk_sweep_has_matched_runs() -> None:
    root = CONFIG_DIR.parents[2]
    spec = load_spec(path=root / "configs/sweep/ph+action-mode+k.toml")
    group_dir = root / spec["group"]
    baseline = read_config_file(path=group_dir / "BASE.toml")
    plan = build_plan(spec=spec, group_dir=group_dir, base_config=baseline)
    grid = spec["grid"]
    assert set(grid) == {"action_mode", "action_chunk_size"}
    assert set(grid["action_mode"]) == {JOINT, OSC}
    expected = {
        (mode, k)
        for mode in grid["action_mode"]
        for k in grid["action_chunk_size"]
    }
    assert len(plan) == len(expected)
    assert len({entry["slug"] for entry in plan}) == len(expected)
    assert {
        (entry["effective"]["action_mode"], entry["effective"]["action_chunk_size"])
        for entry in plan
    } == expected
    for entry in plan:
        config = entry["effective"]
        assert config == {
            **baseline,
            "action_mode": config["action_mode"],
            "action_chunk_size": config["action_chunk_size"],
            "rollout": {"episodes": 30, "horizon": 200, "seed": 42, "n_action_steps": [10]},
            "lr": 1e-5,
            "steps": 100000,
            "use_z": True,
        }
        assert config["dataset"] == OSC_DATASET
        assert config["rollout"]["n_action_steps"] == [10]
