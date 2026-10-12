"""Run the posterior-collapse audit suite for a saved sweep iteration or one run.

`--iteration` enumerates every run recorded by the iteration; `--run` targets a
single run by the TensorBoard run name it shows (`<combo-id>/<run-name>`, or the
bare `<run-name>`). Each run resolves its own latest checkpoint and dispatches
each requested test to its runner and plot script:

    uv run scripts/utils/posterior-collpase/run_suite.py --iteration i526a92dd6741 --tests 1 2 3 4
    uv run scripts/utils/posterior-collpase/run_suite.py --run 0000/2026-10-12_10-36-35_slug-i526a92dd6741

`--step` pins one checkpoint step for every run instead of using each run's
latest. `--tests` defaults to every known test id. Tests 2 and 3 have no runner
yet, and test 4 is implemented only by the checkpoint-pinned `run_test4.py`, so
both are reported and skipped rather than silently ignored.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout

CHECKS_ROOT = Path('checkpoints/act_v2')
ALL_TESTS = (1, 2, 3, 4)
UNIMPLEMENTED = (2, 3)
# TensorBoard publishes each run as <combo-id>/<run-name>; the run name always
# ends in the owning iteration's `-i<12 hex>` token.
RUN_REFERENCE = re.compile(r'^(?:(?P<combo_id>[0-9]{4})/)?(?P<run_name>[^/]+)$')
RUN_TOKEN = re.compile(r'-i?([0-9a-f]{12})$')


def find_run_directory(*, run_name: str, root: Path) -> Path:
    """The newest local checkpoint directory whose name ends with `_<run_name>`."""
    matches = sorted(path for path in root.iterdir()
                     if path.is_dir() and path.name.endswith('_' + run_name))
    if not matches:
        raise FileNotFoundError(f'no local checkpoint directory ends with _{run_name}')
    return matches[-1]


def parse_run_reference(*, value: str) -> tuple[str, str] | None:
    """`(combo_id, run_name)` for a TensorBoard run name, or None when malformed.

    Accepts `<combo-id>/<run-name>` and the bare `<run-name>`; both require the
    trailing `-i<12 hex>` iteration token that names the owning sweep.
    """
    match = RUN_REFERENCE.fullmatch(value)
    if match is None or RUN_TOKEN.search(match['run_name']) is None:
        return None
    return match['combo_id'] or '', match['run_name']


def resolve_run_scope(*, value: str, iteration_root: Path, checks_root: Path) -> tuple[str, dict, Path]:
    """Resolve a TensorBoard run name to its iteration token, combo, and checkpoint directory."""
    parsed = parse_run_reference(value=value)
    if parsed is None:
        raise ValueError(f'run name {value!r} must be <combo-id>/<run-name> or a bare <run-name>, '
                         'ending in the iteration token -i<12 hex>')
    combo_id, run_name = parsed
    iteration = 'i' + RUN_TOKEN.search(run_name).group(1)
    iteration_dir = layout.locate_iteration(iteration=iteration, root=iteration_root)
    if iteration_dir is None:
        raise ValueError(f'no saved iteration matches {iteration!r} under {iteration_root}')
    combos = json.loads((iteration_dir / 'manifest.json').read_text())['combos']
    if combo_id:
        matched = [combo for combo in combos if combo['id'] == combo_id]
    else:
        matched = [combo for combo in combos if run_name.endswith(combo['name'])]
    if len(matched) != 1:
        raise ValueError(f'run name {value!r} matches {len(matched)} combos in {iteration_dir}')
    run_directory = Path(checks_root) / run_name
    if not run_directory.is_dir():
        raise FileNotFoundError(f'checkpoint directory not found: {run_directory}')
    return iteration, matched[0], run_directory


def select_checkpoint(*, directory: Path, step: str | None) -> Path:
    """The named checkpoint, or the highest-numbered `step_*.pt` when step is None."""
    if step is not None:
        checkpoint = directory / f'{step}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{checkpoint} not found')
        return checkpoint
    checkpoints = sorted(directory.glob('step_*.pt'), key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    if not checkpoints:
        raise FileNotFoundError(f'no step_*.pt under {directory}')
    return checkpoints[-1]


def parse_tests(*, values: list[str]) -> list[int]:
    """Validated test ids, all of them when none are given."""
    tests = [int(value) for value in values] if values else list(ALL_TESTS)
    unknown = sorted(set(tests) - set(ALL_TESTS))
    if unknown:
        raise ValueError(f'unknown test ids {unknown}; known ids are {list(ALL_TESTS)}')
    return sorted(set(tests))


def pinned_checkpoint() -> Path:
    """The checkpoint that the checkpoint-pinned test-4 runner uses."""
    import run_test4
    return Path(run_test4.CHECKPOINT)


def run(*, command: list[str], dry_run: bool) -> None:
    print('+ ' + ' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--iteration', help='saved iteration token, e.g. i526a92dd6741 (every run)')
    target.add_argument('--run', help='TensorBoard run name, e.g. 0000/2026-10-12_10-36-35_slug-i526a92dd6741 '
                                      '(one run; the bare run-name is accepted too)')
    parser.add_argument('--tests', nargs='*', default=[], help='test ids to run (default: all)')
    parser.add_argument('--step', default=None,
                        help="checkpoint step pinned for every run (default: each run's latest)")
    parser.add_argument('--dry-run', action='store_true', help='print commands without running them')
    args = parser.parse_args()

    try:
        tests = parse_tests(values=args.tests)
    except ValueError as error:
        raise SystemExit(str(error))

    run_directory, selected_combo = None, None
    if args.run is not None:
        try:
            iteration, selected_combo, run_directory = resolve_run_scope(
                value=args.run, iteration_root=layout.ITERATION_ROOT, checks_root=CHECKS_ROOT)
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(str(error))
    else:
        iteration = args.iteration
    iteration_dir = layout.locate_iteration(iteration=iteration)
    if iteration_dir is None:
        raise SystemExit(f'no saved iteration matches {iteration!r} under {layout.ITERATION_ROOT}')
    manifest = json.loads((iteration_dir / 'manifest.json').read_text())
    train_commit = manifest.get('git_commit', '')
    combos = [selected_combo] if selected_combo is not None else manifest['combos']

    skipped, failed = [], []
    for combo in combos:
        run_name = combo['name']
        try:
            run_dir = run_directory if run_directory is not None else find_run_directory(
                run_name=run_name, root=CHECKS_ROOT)
            checkpoint = select_checkpoint(directory=run_dir, step=args.step)
        except FileNotFoundError as error:
            print(f'SKIP {run_name}: {error}', flush=True)
            skipped.append((run_name, str(error)))
            continue
        tag = layout.tag_for_run(iteration=iteration, run_name=run_name)
        output = layout.checkpoint_directory(output_root=layout.OUTPUT_ROOT, iteration=iteration,
                                             checkpoint_stem=checkpoint.stem, tag=tag)
        label = f'{run_name} [{tag or "no tag"}] @ {checkpoint.stem}'
        print(f'RUN {label}', flush=True)
        for test in tests:
            try:
                if test == 1:
                    command = ['uv', 'run', 'scripts/utils/posterior-collpase/run_test1.py',
                               '--iteration', iteration, '--checkpoint', str(checkpoint)]
                    if train_commit:
                        command += ['--train-commit', train_commit]
                    run(command=command, dry_run=args.dry_run)
                    run(command=['uv', 'run', 'scripts/utils/posterior-collpase/plot_test1.py',
                                 '--checkpoint-dir', str(output)], dry_run=args.dry_run)
                elif test in UNIMPLEMENTED:
                    print(f'SKIP {label} test-{test}: no runner implemented', flush=True)
                else:
                    pinned = pinned_checkpoint()
                    if checkpoint.resolve() != pinned.resolve():
                        print(f'SKIP {label} test-4: run_test4.py is pinned to {pinned}', flush=True)
                    else:
                        run(command=['uv', 'run', 'scripts/utils/posterior-collpase/run_test4.py'],
                            dry_run=args.dry_run)
                        run(command=['uv', 'run', 'scripts/utils/posterior-collpase/plot_test4.py'],
                            dry_run=args.dry_run)
            except subprocess.CalledProcessError as error:
                print(f'FAIL {label} test-{test}: exit {error.returncode}', flush=True)
                failed.append((label, test))
    print(f'SUITE done: {len(tests)} test(s) x {len(combos)} run(s); '
          f'{len(skipped)} skipped run(s), {len(failed)} failed test(s)', flush=True)
    if failed or skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
