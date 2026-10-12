"""Action contracts shared by extraction, training, checkpoints and execution.

OSC targets are controller *inputs*, not physical pose increments. Robosuite
performs the scaling once, when executing these commands.
"""
from __future__ import annotations

import copy
import json

import h5py
import numpy as np

JOINT = "joint_position_aperture"
OSC = "osc_pose_delta"
ACTION_DIMS = {JOINT: 8, OSC: 7}
SOURCE_ENV_ATTR = "source_env_args"
EXTRACTION_GUIDANCE = (
    "re-extract from the original demonstrations with scripts/extract_act_dataset.py "
    "(commands cannot be synthesized from next_obs)"
)


def action_dims(*, action_mode: str = JOINT) -> int:
    if not isinstance(action_mode, str) or action_mode not in ACTION_DIMS:
        raise ValueError(f"unknown action_mode {action_mode!r}; expected {list(ACTION_DIMS)}")
    return ACTION_DIMS[action_mode]


def source_env_metadata(*, data: h5py.Group) -> dict:
    raw = data.attrs.get(SOURCE_ENV_ATTR, data.attrs.get("env_args"))
    if raw is None:
        raise ValueError("OSC requires source_env_args or supported legacy env_args metadata")
    try:
        metadata = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid source environment JSON metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("source environment JSON metadata must be an object with env_kwargs")
    return metadata


def osc_arm_config(*, env_meta: dict) -> dict:
    config = env_meta["env_kwargs"]["controller_configs"]
    return config["body_parts"]["right"] if "body_parts" in config else config


def six_axis_bounds(*, arm: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    low_key, high_key = f"{prefix}_min", f"{prefix}_max"
    try:
        low = np.broadcast_to(np.asarray(arm[low_key], dtype=float), (6,))
        high = np.broadcast_to(np.asarray(arm[high_key], dtype=float), (6,))
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"OSC requires explicit six-axis {low_key}/{high_key} scaling") from error
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not (low < high).all():
        raise ValueError(f"invalid OSC {low_key}/{high_key} bounds")
    return low, high


def osc_command_bounds(*, representation: dict) -> tuple[np.ndarray, np.ndarray]:
    arm = osc_arm_config(env_meta=representation["source_env_args"])
    low, high = six_axis_bounds(arm=arm, prefix="input")
    # Panda's GRIP controller accepts a scalar opening/closing command in [-1,1].
    return np.r_[low, -1], np.r_[high, 1]


def verify_osc_metadata(*, env_meta: dict) -> dict:
    """Accept only explicit, fixed-impedance world-frame delta OSC semantics."""
    if not isinstance(env_meta, dict) or not isinstance(env_meta.get("env_kwargs"), dict):
        raise ValueError("OSC requires source env_kwargs metadata")
    if not isinstance(env_meta.get("env_name"), str) or not env_meta["env_name"]:
        raise ValueError("OSC requires an explicit source env_name")
    if env_meta.get("type") != 1:
        raise ValueError("OSC supports only robosuite source metadata (type=1)")
    env_meta = copy.deepcopy(env_meta)
    kwargs = env_meta["env_kwargs"]
    if kwargs.get("robots") not in ("Panda", ["Panda"]):
        raise ValueError("OSC supports only a single Panda source robot; set robots explicitly")
    config = kwargs.get("controller_configs")
    if not isinstance(config, dict):
        raise ValueError("OSC requires source controller_configs metadata")
    if "body_parts" in config:
        if (
            config.get("type") != "BASIC" or not isinstance(config["body_parts"], dict)
            or set(config["body_parts"]) != {"right"}
        ):
            raise ValueError("supported OSC composite controller is BASIC with one right arm")
    try:
        arm = osc_arm_config(env_meta=env_meta)
    except (KeyError, TypeError) as error:
        raise ValueError("OSC requires a source right-arm controller") from error
    if not isinstance(arm, dict):
        raise ValueError("OSC requires a source right-arm controller dictionary")
    required = {"type": "OSC_POSE", "input_ref_frame": "world", "impedance_mode": "fixed"}
    for key, expected in required.items():
        if arm.get(key) != expected:
            raise ValueError(f"unsupported OSC source controller: {key} must be {expected!r}")
    if "body_parts" not in config:
        # Historical single-arm robosuite controllers implicitly used scalar GRIP.
        # Materialize that default, but do not guess interpolation semantics.
        arm.setdefault("gripper", {"type": "GRIP"})
        kwargs["controller_configs"] = {"type": "BASIC", "body_parts": {"right": arm}}
    if not isinstance(arm.get("gripper"), dict) or arm["gripper"].get("type") != "GRIP":
        raise ValueError("OSC requires an explicit scalar GRIP gripper controller")
    if "interpolation" not in arm or arm["interpolation"] not in (None, "linear"):
        raise ValueError("OSC requires explicit interpolation (null or 'linear'); re-extract source metadata")
    if arm["interpolation"] == "linear":
        ramp = arm.get("ramp_ratio")
        if not isinstance(ramp, (int, float)) or isinstance(ramp, bool) or not 0 < ramp <= 1:
            raise ValueError("linear OSC interpolation requires explicit ramp_ratio in (0, 1]")
    # robosuite 1.5 uses input_type; legacy sources use control_delta. Require
    # positive evidence of delta semantics, and reject conflicting declarations.
    if arm.get("input_type") != "delta" and arm.get("control_delta") is not True:
        raise ValueError("OSC source controller must explicitly specify delta inputs")
    if arm.get("input_type", "delta") != "delta" or arm.get("control_delta", True) is not True:
        raise ValueError("conflicting OSC delta metadata")
    six_axis_bounds(arm=arm, prefix="input")
    six_axis_bounds(arm=arm, prefix="output")
    frequency = kwargs.get("control_freq")
    if (
        not isinstance(frequency, (int, float)) or isinstance(frequency, bool)
        or not np.isfinite(frequency) or frequency <= 0
    ):
        raise ValueError("OSC requires explicit positive control_freq")
    return copy.deepcopy(env_meta)


def representation_metadata(*, action_mode: str = JOINT, source_env: dict | None = None) -> dict:
    result = {
        "mode": action_mode,
        "action_dims": action_dims(action_mode=action_mode),
        "proprio_dims": 8,
    }
    if action_mode == OSC:
        if source_env is None:
            raise ValueError("OSC representation requires verified source metadata")
        result["source_env_args"] = verify_osc_metadata(env_meta=source_env)
        result["semantics"] = "source_controller_inputs"
    else:
        result["semantics"] = "achieved_next_joint_position_and_aperture"
    return result


def resolve_checkpoint_representation(*, checkpoint: dict) -> dict:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a dictionary")
    mode = config.get("action_mode", JOINT)
    dimensions = {"action_dims": action_dims(action_mode=mode), "proprio_dims": 8}
    for key, expected in dimensions.items():
        if key in config and config[key] != expected:
            raise ValueError(f"checkpoint config {key} disagrees with action_mode {mode!r}: expected {expected}")
    stored = checkpoint.get("action_representation")
    if stored is None:
        if mode != JOINT:
            raise ValueError("OSC checkpoint missing action_representation/source semantics")
        return representation_metadata()
    if not isinstance(stored, dict):
        raise ValueError("checkpoint action_representation must be a dictionary")
    if stored.get("mode") != mode:
        raise ValueError("checkpoint action_mode disagrees with action_representation")
    expected = representation_metadata(action_mode=mode, source_env=stored.get("source_env_args"))
    for key in ("action_dims", "proprio_dims", "semantics"):
        if stored.get(key) != expected[key]:
            raise ValueError(f"checkpoint representation mismatch: {key}")
    return expected


def validate_normalization(*, stats: dict, action_mode: str = JOINT) -> None:
    for key in ("proprio_mean", "proprio_std", "action_mean", "action_std"):
        size = 8 if key.startswith("proprio") else action_dims(action_mode=action_mode)
        value = np.asarray(stats[key], dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"normalization {key} must be finite with shape ({size},)")
        if key.endswith("std") and not (value > 0).all():
            raise ValueError(f"normalization {key} must be positive")


def target_actions(
    *, demo: h5py.Group, action_mode: str = JOINT,
    start: int = 0, stop: int | None = None,) -> np.ndarray:
    if action_mode == OSC:
        return demo["actions"][start:stop].astype(np.float32)
    action_dims(action_mode=action_mode)
    joint = demo["next_obs/robot0_joint_pos"][start:stop]
    gripper = demo["next_obs/robot0_gripper_qpos"][start:stop]
    return np.concatenate([joint, gripper[..., :1] - gripper[..., 1:2]], axis=-1).astype(np.float32)


def validate_numeric_array(*, demo: h5py.Group, path: str, shape: tuple[int, ...]) -> None:
    if path not in demo:
        raise ValueError(f"demo {demo.name} missing {path}")
    ds = demo[path]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"demo {demo.name}/{path}: expected numeric dataset")
    if ds.shape != shape or ds.dtype.kind not in "fiu" or not np.isfinite(ds[:]).all():
        raise ValueError(
            f"demo {demo.name}/{path}: expected finite numeric array {shape}, got {ds.shape}"
        )


def preflight_dataset(
    *, hdf5: h5py.File, image_keys: tuple[str, ...], action_mode: str = JOINT,) -> dict:
    """Validate every demo before allocating a model or starting a training run."""
    action_dims(action_mode=action_mode)
    if "data" not in hdf5 or not len(hdf5["data"]):
        raise ValueError("dataset contains no demos under data/")
    if not image_keys:
        raise ValueError("image_keys must contain at least one observation key")
    data = hdf5["data"]
    if action_mode == OSC:
        for name, demo in data.items():
            if "actions" not in demo:
                raise ValueError(f"demo {name} missing actions; {EXTRACTION_GUIDANCE}")
    source_env = source_env_metadata(data=data) if action_mode == OSC else None
    representation = representation_metadata(action_mode=action_mode, source_env=source_env)
    image_shapes = {}
    for name, demo in data.items():
        n = demo.attrs.get("num_samples")
        if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0:
            raise ValueError(f"demo {name}: num_samples must be a positive integer")
        paths = {"obs/robot0_joint_pos": (n, 7), "obs/robot0_gripper_qpos": (n, 2)}
        if action_mode == JOINT:
            paths.update({
                "next_obs/robot0_joint_pos": (n, 7),
                "next_obs/robot0_gripper_qpos": (n, 2),
            })
        else:
            paths["actions"] = (n, 7)
        for path, shape in paths.items():
            validate_numeric_array(demo=demo, path=path, shape=shape)
        for key in image_keys:
            path = f"obs/{key}"
            if path not in demo:
                raise ValueError(f"demo {name} is missing {path}")
            ds = demo[path]
            if (
                not isinstance(ds, h5py.Dataset) or ds.ndim != 4
                or ds.shape[0] != n or ds.shape[-1] != 3
                or min(ds.shape[1:3]) <= 0 or ds.dtype.kind not in "fiu"
            ):
                raise ValueError(f"demo {name}/{path}: expected numeric (T,H,W,3) images")
            if key in image_shapes and image_shapes[key] != ds.shape[1:]:
                raise ValueError(f"demo {name}/{path}: inconsistent image shape")
            image_shapes[key] = ds.shape[1:]
            for start in range(0, n, 64):
                if not np.isfinite(ds[start:start + 64]).all():
                    raise ValueError(f"demo {name}/{path}: nonfinite images")
        if action_mode == OSC:
            lower, upper = osc_command_bounds(representation=representation)
            actions = demo["actions"][:]
            if np.any(actions < lower - 1e-6) or np.any(actions > upper + 1e-6):
                raise ValueError(f"demo {name}/actions outside source controller input bounds")
    return representation


def bounded_osc_command(*, action: np.ndarray, representation: dict) -> np.ndarray:
    value = np.asarray(action)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("OSC command must be finite with shape (7,)")
    lower, upper = osc_command_bounds(representation=representation)
    return np.clip(value, lower, upper).astype(np.float32)
