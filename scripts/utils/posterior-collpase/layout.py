"""Output layout for the posterior-collapse audit artifacts.

A checkpoint's directory is

    <output_root>/<iteration>/[<config_tag>/]<checkpoint_stem>/

The `iteration` token (`i<12 hex>`) names a saved sweep under
`.vast-train-local/ablations/`. Every run in that sweep shares the token, and
runs can also share a checkpoint stem, so `config_tag` names only the sweep's
varying `[grid]` parameters: each rendered as the run-name label followed by its
value (`k50`, `beta0.1`, `z1`) and joined with `+`. A sweep with no varying
parameter (every `[grid]` list of length 1) has no tag and keeps the flat
`<iteration>/<checkpoint_stem>/` path.
"""
from pathlib import Path
import json
import re
import sys
import tomllib

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.action_modes import JOINT
from scripts.train_v2 import (
    RUN_NAME_FIELDS,
    format_config_value,
    format_dataset,
    format_image_keys,
    sanitize_slug,
)

ITERATION_ROOT = REPO / '.vast-train-local' / 'ablations'
OUTPUT_ROOT = Path('debug/posterior-collapse')
SHORT_ITERATION = re.compile(r'i?[0-9a-f]{12}')


def locate_iteration(*, iteration: str) -> Path | None:
    """The saved iteration directory for an `i<12 hex>` token, if unique."""
    if not SHORT_ITERATION.fullmatch(iteration):
        return None
    token = iteration[1:] if iteration.startswith('i') else iteration
    matches = sorted(ITERATION_ROOT.glob(f'*/*-{token}'))
    return matches[0] if len(matches) == 1 else None


def sweep_grid(*, iteration: str) -> dict | None:
    """The `[grid]` of the sweep that produced `iteration`.

    `None` means the saved iteration could not be located; `{}` means it was
    located but has no grid (a single-combination sweep).
    """
    directory = locate_iteration(iteration=iteration)
    if directory is None:
        return None
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('kind') != 'sweep':
        return {}
    source = directory / manifest['inputs'][0]['path']
    return tomllib.loads(source.read_text()).get('grid', {})


def varying_keys(*, grid: dict) -> list[str]:
    """Grid parameters that take more than one value in the sweep."""
    return [key for key, values in grid.items()
            if len({tuple(value) if isinstance(value, list) else value for value in values}) > 1]


def _token(*, key: str, config: dict) -> str | None:
    """Run-name token for one grid parameter, or None when it has no token."""
    value = config.get(key)
    if key == 'action_mode':
        return None if not value or value == JOINT else sanitize_slug(text=str(value))
    fields = {name: (label, template) for name, label, template in RUN_NAME_FIELDS}
    if key not in fields:
        return f'{sanitize_slug(text=key)}{sanitize_slug(text=str(value))}'
    label, template = fields[key]
    if key == 'image_keys':
        rendered = format_image_keys(image_keys=value)
    elif key == 'dataset':
        rendered = format_dataset(dataset=value)
    else:
        rendered = format_config_value(value=value, template=template)
    return f'{sanitize_slug(text=label)}{rendered}'


def config_tag(*, config: dict, varying: list[str]) -> str:
    """`+`-joined tokens for the varying parameters, in run-name order."""
    known = [name for name, _, _ in RUN_NAME_FIELDS]
    ordered = ([key for key in ('action_mode',) if key in varying]
               + [key for key in known if key in varying]
               + sorted(key for key in varying if key not in known and key != 'action_mode'))
    return '+'.join(token for key in ordered if (token := _token(key=key, config=config)))


def resolve_tag(*, iteration: str, config: dict, sweep: Path | None = None, tag: str | None = None) -> str:
    """The config tag for one checkpoint, from an explicit tag, a sweep spec, or saved iteration."""
    if tag is not None:
        return tag
    grid = tomllib.loads(Path(sweep).read_text()).get('grid', {}) if sweep else sweep_grid(iteration=iteration)
    if not grid:
        return ''
    return config_tag(config=config, varying=varying_keys(grid=grid))


def tag_for_run(*, iteration: str, run_name: str) -> str:
    """The config tag for one run of a saved iteration, keyed by its run name."""
    directory = locate_iteration(iteration=iteration)
    if directory is None:
        return ''
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('kind') != 'sweep':
        return ''
    grid = tomllib.loads((directory / manifest['inputs'][0]['path']).read_text()).get('grid', {})
    if not grid:
        return ''
    for combo in manifest['combos']:
        if run_name.endswith(combo['name']):
            config = tomllib.loads((directory / combo['config']).read_text())
            return config_tag(config=config, varying=varying_keys(grid=grid))
    return ''


def checkpoint_directory(*, output_root: Path, iteration: str, checkpoint_stem: str, tag: str) -> Path:
    """The `<iteration>/[<tag>/]<checkpoint_stem>` directory for one checkpoint."""
    directory = Path(output_root) / iteration
    if tag:
        directory = directory / tag
    return directory / checkpoint_stem


def assert_checkpoint(*, directory: Path, checkpoint_sha: str) -> None:
    """Fail closed when a directory already holds a different checkpoint's artifacts."""
    marker = Path(directory) / 'checkpoint.sha256'
    if not marker.exists():
        return
    recorded = marker.read_text().split()[0]
    if recorded != checkpoint_sha:
        raise RuntimeError(
            f'{Path(directory)} already holds checkpoint {recorded[:16]}, not {checkpoint_sha[:16]}; '
            'refusing to overwrite. Use a distinct --tag/--iteration or remove the stale directory.')
