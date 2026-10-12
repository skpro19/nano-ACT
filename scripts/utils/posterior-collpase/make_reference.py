"""Create the reference template tree for posterior-collapse audit artifacts.

The tree under `debug/posterior-collapse/reference/` is a tiny, valid stub of
every required artifact, populated for all four `i526a92dd6741` config tags. It
is documentation, not results: replace the stubs when producing a real audit.
Run from the repository root:

    uv run python scripts/utils/posterior-collpase/make_reference.py
"""
from pathlib import Path
import json
import struct
import zlib

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'debug' / 'posterior-collapse' / 'reference'
ITERATION = 'i526a92dd6741'
STEM = 'step_000100000'
TAGS = ('k50+beta0.1', 'k50+beta10', 'k100+beta0.1', 'k100+beta10')
SPLITS = ('train', 'held-out-candidate')
SEEDS = (42, 123)

TEST1_ROOT = ('checkpoint.sha256', 'compatibility.json', 'provenance.json', 'results.json', 'summary.md')
TEST1_SEED = ('arrays.npz', 'metadata.json', '01_sd_per_dim.png', '02_mu_histograms.png',
              '03_matched_pairs.png', '04_kl_to_prior.png', '05_per_demo_mean_mu.png')
TEST4_ROOT = ('provenance.json', 'environment.txt', 'results.json', 'summary.md',
              'execution.log', 'plot_execution.log', 'artifact_validation.log', 'execution_notes.md')
TEST4_SEED = ('arrays.npz', 'metadata.json', 'results.json', 'summary.md',
              '01_decoder_latent_sensitivity.png', 'plot_provenance.json')
# Tests 2 and 3 have no runner or plot script yet, so only the generic artifacts exist.
GENERIC_ROOT = ('provenance.json', 'results.json', 'summary.md')
GENERIC_SEED = ('arrays.npz', 'metadata.json')
PLACEHOLDER = 'Reference placeholder — replace with the real artifact.\n'

README = """\
# Posterior-collapse reference layout

Tiny valid stubs of every required artifact, populated for each config tag of
`i526a92dd6741`. **This is documentation, not results**: the `.json`, `.png` and
`.npz` files are placeholders and must be replaced by a real run.

The tree mirrors `debug/posterior-collapse/<iteration>/[<config-tag>/]<stem>/`,
where `<config-tag>` names the sweep's varying `[grid]` parameters (here
`k50/k100` and `beta0.1/beta10`) and is omitted for a single-combination sweep.
See `notes/posterior-tests.md` for the rules and the per-file description.

- `test-1/` — posterior variation; five plots and one `arrays.npz`/`metadata.json`
  pair per `{split}/seed-{seed}/`.
- `test-2/`, `test-3/` — no runner or plot script yet, so only the generic
  `provenance.json`, `results.json`, `summary.md` and per-seed arrays exist.
- `test-4/` — decoder sensitivity; one plot and `plot_provenance.json` per
  `{split}/seed-{seed}/`, plus the execution logs at the test root.
- Regenerate with `uv run python scripts/utils/posterior-collpase/make_reference.py`.
"""


def png_bytes() -> bytes:
    """A minimal valid 1x1 transparent PNG."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    header = struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)
    pixel = b'\x00' + b'\x00\x00\x00\x00'
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
            + chunk(b'IDAT', zlib.compress(pixel)) + chunk(b'IEND', b''))


def write_stub(*, path: Path) -> None:
    """Write one tiny valid file whose name matches the required artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.png':
        path.write_bytes(png_bytes())
    elif path.suffix == '.npz':
        # Local import keeps the header light; numpy writes a valid archive.
        import numpy as np
        np.savez(path, placeholder=np.zeros(1, dtype=np.float32))
    elif path.suffix == '.json':
        path.write_text(json.dumps({'_reference': True, 'placeholder': path.name}, indent=2) + '\n')
    else:
        path.write_text(PLACEHOLDER)


def populate_test(*, directory: Path, root_files, seed_files) -> None:
    for name in root_files:
        write_stub(path=directory / name)
    for split in SPLITS:
        for seed in SEEDS:
            for name in seed_files:
                write_stub(path=directory / split / f'seed-{seed}' / name)


def main() -> None:
    for tag in TAGS:
        base = ROOT / ITERATION / tag / STEM
        populate_test(directory=base / 'test-1', root_files=TEST1_ROOT, seed_files=TEST1_SEED)
        populate_test(directory=base / 'test-2', root_files=GENERIC_ROOT, seed_files=GENERIC_SEED)
        populate_test(directory=base / 'test-3', root_files=GENERIC_ROOT, seed_files=GENERIC_SEED)
        populate_test(directory=base / 'test-4', root_files=TEST4_ROOT, seed_files=TEST4_SEED)
    (ROOT / 'README.md').write_text(README)
    print(f'wrote reference tree under {ROOT}')


if __name__ == '__main__':
    main()
