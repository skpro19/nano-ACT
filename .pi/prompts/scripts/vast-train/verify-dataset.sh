#!/usr/bin/env bash
set -euo pipefail
cd /workspace/toy-act
/root/.local/bin/uv run --frozen --only-group train python - <<'PY'
import hashlib
import os
from pathlib import Path
import h5py
from scripts.train_v2 import load_config
from scripts.rollout import camera_names_from_image_keys
try:
    from scripts.action_modes import JOINT, preflight_dataset
except ImportError as error:
    raise RuntimeError('Pinned code lacks scripts.action_modes.preflight_dataset; publish a compatible revision') from error
config = load_config(path=Path('.vast-train/train-config.toml'))
camera_names_from_image_keys(image_keys=tuple(config['image_keys']))
path = Path(config['dataset'])
assert path.stat().st_size == int(os.environ['EXPECTED_BYTES']), 'Dataset byte size mismatch'
if os.environ.get('EXPECTED_SHA'):
    checksum = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            checksum.update(block)
    assert checksum.hexdigest() == os.environ['EXPECTED_SHA'], 'Dataset SHA-256 mismatch'
with h5py.File(path, 'r') as dataset:
    preflight_dataset(hdf5=dataset, image_keys=tuple(config['image_keys']),
                      action_mode=config.get('action_mode', JOINT))
print('Dataset size/hash/cameras/action representation verified')
PY
