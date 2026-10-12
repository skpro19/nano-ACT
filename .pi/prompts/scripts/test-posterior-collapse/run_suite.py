"""Run the posterior-collapse audit suite for a saved sweep iteration or one run.

`--iteration` enumerates every run recorded by the iteration; `--run` targets a
single run by the TensorBoard run name it shows (`<combo-id>/<run-name>`, or the
bare `<run-name>`). The sweep's checkpoints live in S3
(`s3://toy-act/checkpoints/act_v2` by default), which is the source of truth:
local `checkpoints/act_v2` mirrors can be partial, so each run's latest step is
resolved against the bucket and downloaded into the local mirror before its
tests run.

    uv run .pi/prompts/scripts/test-posterior-collapse/run_suite.py --iteration i526a92dd6741 --tests 1 2 3 4
    uv run .pi/prompts/scripts/test-posterior-collapse/run_suite.py --run 0000/2026-10-12_10-36-35_slug-i526a92dd6741

`--step` pins one checkpoint step for every run instead of using each run's
latest. `--tests` defaults to every known test id. Tests 2–4 reuse verified
test-1 arrays; the prerequisite is prepared automatically when absent.
"""
from pathlib import Path
from typing import Any
import argparse
import json
import os
import re
import subprocess
import sys

import boto3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout

CHECKPOINT_MIRROR = Path('checkpoints/act_v2')
BUCKET = os.environ.get('TOY_ACT_BUCKET', 'toy-act')
BUCKET_CHECKPOINT_PREFIX = os.environ.get('TOY_ACT_CHECKPOINT_PREFIX', 'checkpoints/act_v2')
BUCKET_REGION = os.environ.get('TOY_ACT_AWS_REGION', 'ap-south-1')
BUCKET_PROFILE = os.environ.get('TOY_ACT_AWS_PROFILE', 'toy-pickplace-backup')
ALL_TESTS = (1, 2, 3, 4)
STEP_CHECKPOINT = re.compile(r'step_(\d+)\.pt')
# TensorBoard publishes each run as <combo-id>/<run-name>; the run name always
# ends in the owning iteration's `-i<12 hex>` token.
RUN_REFERENCE = re.compile(r'^(?:(?P<combo_id>[0-9]{4})/)?(?P<run_name>[^/]+)$')
RUN_TOKEN = re.compile(r'-i?([0-9a-f]{12})$')


def checkpoint_step(*, name: str) -> int:
    """The numeric step in a `step_<digits>.pt` checkpoint name."""
    match = STEP_CHECKPOINT.fullmatch(name)
    if match is None:
        raise ValueError(f'not a checkpoint name: {name}')
    return int(match.group(1))


class CheckpointBucket:
    """Runs and checkpoints in the sweep's S3 bucket, the source of truth.

    The bucket is authoritative and local `checkpoints/act_v2` mirrors can miss
    steps, so every run and checkpoint name is resolved here and only then
    downloaded into the local mirror for the test runners to load.
    """

    def __init__(self, *, client: Any, bucket: str, prefix: str) -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.rstrip('/')
        self._run_directories: list[str] | None = None

    def _list_v2(self, *, prefix: str, delimiter: str | None = None) -> list[dict[str, Any]]:
        """Every response page of one `list_objects_v2` query, continuation included."""
        pages = []
        token = None
        while True:
            request: dict[str, Any] = {'Bucket': self._bucket, 'Prefix': prefix}
            if delimiter is not None:
                request['Delimiter'] = delimiter
            if token is not None:
                request['ContinuationToken'] = token
            page = self._client.list_objects_v2(**request)
            pages.append(page)
            if not page.get('IsTruncated'):
                return pages
            token = page['NextContinuationToken']

    def run_directories(self) -> list[str]:
        """Every run directory directly under the checkpoint prefix (cached)."""
        if self._run_directories is None:
            base = self._prefix + '/'
            directories = []
            for page in self._list_v2(prefix=base, delimiter='/'):
                directories.extend(item['Prefix'].removeprefix(base).rstrip('/')
                                   for item in page.get('CommonPrefixes', []))
            self._run_directories = sorted(directories)
        return self._run_directories

    def find_run_directory(self, *, run_name: str) -> str:
        """The newest bucket run directory whose name ends with `_<run_name>`."""
        matches = [directory for directory in self.run_directories()
                   if directory.endswith('_' + run_name)]
        if not matches:
            raise FileNotFoundError(f'no bucket run directory ends with _{run_name}')
        return matches[-1]

    def require_run_directory(self, *, run_directory: str) -> None:
        """Fail when the named run directory is not present in the bucket."""
        if run_directory not in self.run_directories():
            raise FileNotFoundError(f'bucket run directory not found: {run_directory}')

    def checkpoint_names(self, *, run_directory: str) -> list[str]:
        """Every `step_*.pt` object under one run directory, lowest step first."""
        prefix = f'{self._prefix}/{run_directory}/'
        names = []
        for page in self._list_v2(prefix=prefix):
            names.extend(item['Key'].removeprefix(prefix)
                         for item in page.get('Contents', [])
                         if STEP_CHECKPOINT.fullmatch(item['Key'].removeprefix(prefix)))
        return sorted(names, key=lambda name: checkpoint_step(name=name))

    def select_checkpoint(self, *, run_directory: str, step: str | None) -> str:
        """The named checkpoint, or the highest-numbered `step_*.pt` when step is None."""
        names = self.checkpoint_names(run_directory=run_directory)
        if step is not None:
            name = f'{step}.pt'
            if name not in names:
                raise FileNotFoundError(f'{self._prefix}/{run_directory}/{name} not found in {self._bucket}')
            return name
        if not names:
            raise FileNotFoundError(f'no step_*.pt under {self._prefix}/{run_directory} in {self._bucket}')
        return names[-1]

    def ensure_local(self, *, run_directory: str, checkpoint_name: str, mirror_root: Path,
                     download: bool = True) -> Path:
        """The local mirror path for one checkpoint, downloading it when missing.

        A local file whose size matches the bucket object is reused; anything else
        is downloaded to a sibling `.tmp` and atomically replaced, so a partial
        transfer is never mistaken for a usable checkpoint.
        """
        key = f'{self._prefix}/{run_directory}/{checkpoint_name}'
        destination = Path(mirror_root) / run_directory / checkpoint_name
        if not download:
            return destination
        expected_size = self._client.head_object(Bucket=self._bucket, Key=key)['ContentLength']
        if destination.is_file() and destination.stat().st_size == expected_size:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        self._client.download_file(self._bucket, key, str(temporary))
        temporary.replace(destination)
        return destination


def build_bucket_client(*, profile: str | None, region: str) -> Any:
    """A boto3 S3 client using `profile` when given and the default chain otherwise."""
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client('s3')


def parse_run_reference(*, value: str) -> tuple[str, str] | None:
    """`(combo_id, run_name)` for a TensorBoard run name, or None when malformed.

    Accepts `<combo-id>/<run-name>` and the bare `<run-name>`; both require the
    trailing `-i<12 hex>` iteration token that names the owning sweep.
    """
    match = RUN_REFERENCE.fullmatch(value)
    if match is None or RUN_TOKEN.search(match['run_name']) is None:
        return None
    return match['combo_id'] or '', match['run_name']


def resolve_run_scope(*, value: str, iteration_root: Path,
                      bucket: CheckpointBucket) -> tuple[str, dict, str]:
    """Resolve a TensorBoard run name to its iteration token, combo, and bucket run directory."""
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
    bucket.require_run_directory(run_directory=run_name)
    return iteration, matched[0], run_name


def parse_tests(*, values: list[str]) -> list[int]:
    """Validated test ids, all of them when none are given."""
    tests = [int(value) for value in values] if values else list(ALL_TESTS)
    unknown = sorted(set(tests) - set(ALL_TESTS))
    if unknown:
        raise ValueError(f'unknown test ids {unknown}; known ids are {list(ALL_TESTS)}')
    return sorted(set(tests))


def dispatch_tests(*, tests: list[int], output: Path) -> list[int]:
    """Prepend the sampling prerequisite when later tests lack test-1 results."""
    if any(test > 1 for test in tests) and 1 not in tests and not (output / 'test-1/results.json').exists():
        return [1, *tests]
    return tests


def run(*, command: list[str], dry_run: bool) -> None:
    print('+ ' + ' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def plot_command(*, test: int, output: Path) -> list[str]:
    return ['uv', 'run', f'.pi/prompts/scripts/test-posterior-collapse/plot_test{test}.py',
            '--checkpoint-dir', str(output)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--iteration', help='saved iteration token, e.g. i526a92dd6741 (every run)')
    target.add_argument('--run', help='TensorBoard run name, e.g. 0000/2026-10-12_10-36-35_slug-i526a92dd6741 '
                                      '(one run; the bare run-name is accepted too)')
    parser.add_argument('--tests', nargs='*', default=[], help='test ids to run (default: all)')
    parser.add_argument('--step', default=None,
                        help="checkpoint step pinned for every run, resolved in the bucket (default: each run's latest)")
    parser.add_argument('--dry-run', action='store_true', help='print commands without running them')
    args = parser.parse_args()

    try:
        tests = parse_tests(values=args.tests)
    except ValueError as error:
        raise SystemExit(str(error))

    bucket = CheckpointBucket(client=build_bucket_client(profile=BUCKET_PROFILE or None, region=BUCKET_REGION),
                              bucket=BUCKET, prefix=BUCKET_CHECKPOINT_PREFIX)

    run_directory, selected_combo = None, None
    if args.run is not None:
        try:
            iteration, selected_combo, run_directory = resolve_run_scope(
                value=args.run, iteration_root=layout.ITERATION_ROOT, bucket=bucket)
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
            run_dir = run_directory if run_directory is not None else bucket.find_run_directory(
                run_name=run_name)
            checkpoint_name = bucket.select_checkpoint(run_directory=run_dir, step=args.step)
            checkpoint = bucket.ensure_local(run_directory=run_dir, checkpoint_name=checkpoint_name,
                                             mirror_root=CHECKPOINT_MIRROR, download=not args.dry_run)
        except FileNotFoundError as error:
            print(f'SKIP {run_name}: {error}', flush=True)
            skipped.append((run_name, str(error)))
            continue
        tag = layout.tag_for_run(iteration=iteration, run_name=run_name)
        output = layout.checkpoint_directory(output_root=layout.OUTPUT_ROOT, iteration=iteration,
                                             checkpoint_stem=checkpoint.stem, tag=tag)
        label = f'{run_name} [{tag or "no tag"}] @ {checkpoint.stem}'
        print(f'RUN {label}', flush=True)
        for test in dispatch_tests(tests=tests, output=output):
            try:
                if test == 1:
                    command = ['uv', 'run', '.pi/prompts/scripts/test-posterior-collapse/run_test1.py',
                               '--iteration', iteration, '--checkpoint', str(checkpoint)]
                    if train_commit:
                        command += ['--train-commit', train_commit]
                    run(command=command, dry_run=args.dry_run)
                    run(command=plot_command(test=test, output=output), dry_run=args.dry_run)
                else:
                    command = ['uv', 'run', f'.pi/prompts/scripts/test-posterior-collapse/run_test{test}.py',
                               '--iteration', iteration, '--checkpoint', str(checkpoint)]
                    if train_commit:
                        command += ['--train-commit', train_commit]
                    run(command=command, dry_run=args.dry_run)
                    run(command=plot_command(test=test, output=output), dry_run=args.dry_run)
            except subprocess.CalledProcessError as error:
                print(f'FAIL {label} test-{test}: exit {error.returncode}', flush=True)
                failed.append((label, test))
    print(f'SUITE done: {len(tests)} test(s) x {len(combos)} run(s); '
          f'{len(skipped)} skipped run(s), {len(failed)} failed test(s)', flush=True)
    if failed or skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
