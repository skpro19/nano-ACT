"""Run the posterior-collapse audit suite for one saved sweep iteration.

Enumerate every run recorded by the iteration, resolve each run's own latest
checkpoint, and dispatch each requested test to its runner and plot script:

    uv run scripts/utils/posterior-collpase/run_suite.py --iteration i526a92dd6741 --tests 1 2 3 4

`--step` pins one checkpoint step for every run instead of using each run's
latest. `--tests` defaults to every known test id. Tests 2 and 3 have no runner
yet, and test 4 is implemented only by the checkpoint-pinned `run_test4.py`, so
both are reported and skipped rather than silently ignored.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout

CHECKS_ROOT = Path('checkpoints/act_v2')
ALL_TESTS = (1, 2, 3, 4)
UNIMPLEMENTED = (2, 3)


def find_run_directory(*, run_name: str, root: Path) -> Path:
    """The newest local checkpoint directory whose name ends with `_<run_name>`."""
    matches = sorted(path for path in root.iterdir()
                     if path.is_dir() and path.name.endswith('_' + run_name))
    if not matches:
        raise FileNotFoundError(f'no local checkpoint directory ends with _{run_name}')
    return matches[-1]


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
    parser.add_argument('--iteration', required=True, help='saved iteration token, e.g. i526a92dd6741')
    parser.add_argument('--tests', nargs='*', default=[], help='test ids to run (default: all)')
    parser.add_argument('--step', default=None,
                        help="checkpoint step pinned for every run (default: each run's latest)")
    parser.add_argument('--dry-run', action='store_true', help='print commands without running them')
    args = parser.parse_args()

    try:
        tests = parse_tests(values=args.tests)
    except ValueError as error:
        raise SystemExit(str(error))
    iteration_dir = layout.locate_iteration(iteration=args.iteration)
    if iteration_dir is None:
        raise SystemExit(f'no saved iteration matches {args.iteration!r} under {layout.ITERATION_ROOT}')
    manifest = json.loads((iteration_dir / 'manifest.json').read_text())
    train_commit = manifest.get('git_commit', '')

    skipped, failed = [], []
    for combo in manifest['combos']:
        run_name = combo['name']
        try:
            run_dir = find_run_directory(run_name=run_name, root=CHECKS_ROOT)
            checkpoint = select_checkpoint(directory=run_dir, step=args.step)
        except FileNotFoundError as error:
            print(f'SKIP {run_name}: {error}', flush=True)
            skipped.append((run_name, str(error)))
            continue
        tag = layout.tag_for_run(iteration=args.iteration, run_name=run_name)
        output = layout.checkpoint_directory(output_root=layout.OUTPUT_ROOT, iteration=args.iteration,
                                             checkpoint_stem=checkpoint.stem, tag=tag)
        label = f'{run_name} [{tag or "no tag"}] @ {checkpoint.stem}'
        print(f'RUN {label}', flush=True)
        for test in tests:
            try:
                if test == 1:
                    command = ['uv', 'run', 'scripts/utils/posterior-collpase/run_test1.py',
                               '--iteration', args.iteration, '--checkpoint', str(checkpoint)]
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
    print(f'SUITE done: {len(tests)} test(s) x {len(manifest["combos"])} run(s); '
          f'{len(skipped)} skipped run(s), {len(failed)} failed test(s)', flush=True)
    if failed or skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
