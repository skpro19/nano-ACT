"""Compatibility checks when pairing an analysis dataset with a checkpoint."""
from scripts.action_modes import OSC


def validate_dataset_representation(*, dataset_representation: dict, checkpoint_representation: dict) -> None:
    if dataset_representation["mode"] != checkpoint_representation["mode"]:
        raise ValueError("dataset and checkpoint action modes differ")
    if checkpoint_representation["mode"] == OSC:
        source = dataset_representation["source_env_args"]["env_kwargs"]
        trained = checkpoint_representation["source_env_args"]["env_kwargs"]
        for key in ("controller_configs", "control_freq", "robots", "gripper_types"):
            if source.get(key) != trained.get(key):
                raise ValueError(f"dataset and checkpoint source semantics differ: {key}")
