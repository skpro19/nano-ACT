"""Suite-driver helpers for the posterior-collapse tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / '.pi' / 'prompts' / 'scripts' / 'test-posterior-collapse'))

import run_suite

BUCKET_PREFIX = 'checkpoints/act_v2'


class FakeS3Client:
    """In-memory stand-in for the boto3 S3 client used by CheckpointBucket."""

    def __init__(self, *, objects: dict[str, int]) -> None:
        self.objects = dict(objects)
        self.downloads: list[str] = []

    def list_objects_v2(self, *, Bucket: str, Prefix: str, Delimiter: str | None = None,
                        ContinuationToken: str | None = None) -> dict:
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        if Delimiter is not None:
            prefixes = sorted({key[len(Prefix):].split(Delimiter, 1)[0]
                               for key in keys if Delimiter in key[len(Prefix):]})
            return {'CommonPrefixes': [{'Prefix': f'{Prefix}{name}{Delimiter}'} for name in prefixes]}
        return {'Contents': [{'Key': key, 'Size': self.objects[key]} for key in keys]}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        return {'ContentLength': self.objects[Key]}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append(key)
        Path(filename).write_bytes(b'x' * self.objects[key])


def make_bucket(*, objects: dict[str, int]) -> run_suite.CheckpointBucket:
    return run_suite.CheckpointBucket(client=FakeS3Client(objects=objects), bucket='toy-act',
                                      prefix=BUCKET_PREFIX)


class ParseTestsTests(unittest.TestCase):
    def test_default_is_every_test(self) -> None:
        self.assertEqual(run_suite.parse_tests(values=[]), [1, 2, 3, 4])

    def test_subset_is_sorted_and_deduplicated(self) -> None:
        self.assertEqual(run_suite.parse_tests(values=['2', '1', '2']), [1, 2])

    def test_unknown_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_suite.parse_tests(values=['9'])


class DispatchTestsTests(unittest.TestCase):
    def test_later_tests_add_missing_sampling_prerequisite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(run_suite.dispatch_tests(tests=[2, 3, 4], output=Path(temporary)),
                             [1, 2, 3, 4])

    def test_saved_sampling_results_do_not_duplicate_test_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'test-1').mkdir()
            (output / 'test-1/results.json').write_text('{}')
            self.assertEqual(run_suite.dispatch_tests(tests=[2, 3, 4], output=output), [2, 3, 4])
            self.assertEqual(run_suite.dispatch_tests(tests=[1, 2], output=output), [1, 2])


class ParseRunReferenceTests(unittest.TestCase):
    def test_combo_prefixed_run_name(self) -> None:
        self.assertEqual(
            run_suite.parse_run_reference(value='0000/2026-01-01_00-00-00_slug-i000000000000'),
            ('0000', '2026-01-01_00-00-00_slug-i000000000000'))

    def test_bare_run_name(self) -> None:
        self.assertEqual(
            run_suite.parse_run_reference(value='2026-01-01_00-00-00_slug-i000000000000'),
            ('', '2026-01-01_00-00-00_slug-i000000000000'))

    def test_missing_iteration_token_is_rejected(self) -> None:
        self.assertIsNone(run_suite.parse_run_reference(value='0000/2026-01-01_00-00-00_slug'))

    def test_extra_path_segment_is_rejected(self) -> None:
        self.assertIsNone(run_suite.parse_run_reference(value='0000/extra/2026-01-01_slug-i000000000000'))


class ResolveRunScopeTests(unittest.TestCase):
    def make_context(self, *, directory: Path) -> tuple[Path, run_suite.CheckpointBucket]:
        iteration_root = directory / 'ablations'
        iteration = iteration_root / 'sweep-a' / '20260101T000000Z-aa0000000000'
        iteration.mkdir(parents=True)
        (iteration / 'manifest.json').write_text(json.dumps({'combos': [
            {'id': '0000', 'name': 'slug-iaa0000000000'},
            {'id': '0001', 'name': 'other-iaa0000000000'}]}))
        run = '2026-01-01_00-00-00_slug-iaa0000000000'
        bucket = make_bucket(objects={f'{BUCKET_PREFIX}/{run}/step_000014000.pt': 10})
        return iteration_root, bucket

    def test_combo_id_selects_the_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, bucket = self.make_context(directory=Path(temporary))
            iteration, combo, run_directory = run_suite.resolve_run_scope(
                value='0000/2026-01-01_00-00-00_slug-iaa0000000000',
                iteration_root=iteration_root, bucket=bucket)
            self.assertEqual(iteration, 'iaa0000000000')
            self.assertEqual(combo['id'], '0000')
            self.assertEqual(run_directory, '2026-01-01_00-00-00_slug-iaa0000000000')

    def test_bare_run_name_matches_by_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, bucket = self.make_context(directory=Path(temporary))
            _, combo, run_directory = run_suite.resolve_run_scope(
                value='2026-01-01_00-00-00_slug-iaa0000000000',
                iteration_root=iteration_root, bucket=bucket)
            self.assertEqual(combo['id'], '0000')
            self.assertEqual(run_directory, '2026-01-01_00-00-00_slug-iaa0000000000')

    def test_unknown_combo_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, bucket = self.make_context(directory=Path(temporary))
            with self.assertRaises(ValueError):
                run_suite.resolve_run_scope(
                    value='0002/2026-01-01_00-00-00_slug-iaa0000000000',
                    iteration_root=iteration_root, bucket=bucket)

    def test_unknown_iteration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, bucket = self.make_context(directory=Path(temporary))
            with self.assertRaises(ValueError):
                run_suite.resolve_run_scope(
                    value='0000/2026-01-01_00-00-00_slug-ibb1111111111',
                    iteration_root=iteration_root, bucket=bucket)

    def test_missing_checkpoint_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, bucket = self.make_context(directory=Path(temporary))
            with self.assertRaises(FileNotFoundError):
                run_suite.resolve_run_scope(
                    value='0000/2026-01-01_00-00-00_absent-iaa0000000000',
                    iteration_root=iteration_root, bucket=bucket)


class BucketRunDirectoryTests(unittest.TestCase):
    def test_newest_run_directory_wins(self) -> None:
        bucket = make_bucket(objects={
            f'{BUCKET_PREFIX}/2026-01-01_00-00-00_slug-i000000000000/step_000014000.pt': 10,
            f'{BUCKET_PREFIX}/2026-02-01_00-00-00_slug-i000000000000/step_000014000.pt': 10,
            f'{BUCKET_PREFIX}/unrelated/step_000014000.pt': 10})
        self.assertEqual(bucket.find_run_directory(run_name='slug-i000000000000'),
                         '2026-02-01_00-00-00_slug-i000000000000')

    def test_missing_run_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            make_bucket(objects={}).find_run_directory(run_name='slug-i000000000000')

    def test_required_run_directory_must_exist(self) -> None:
        with self.assertRaises(FileNotFoundError):
            make_bucket(objects={}).require_run_directory(run_directory='absent')


class BucketCheckpointTests(unittest.TestCase):
    def make_bucket(self) -> run_suite.CheckpointBucket:
        self.run_directory = '2026-01-01_00-00-00_slug-i000000000000'
        return make_bucket(objects={
            f'{BUCKET_PREFIX}/{self.run_directory}/step_000008000.pt': 10,
            f'{BUCKET_PREFIX}/{self.run_directory}/step_000014000.pt': 10,
            f'{BUCKET_PREFIX}/{self.run_directory}/step_000100000.pt': 20,
            f'{BUCKET_PREFIX}/{self.run_directory}/last.pt': 10})

    def test_latest_checkpoint_is_the_highest_step(self) -> None:
        bucket = self.make_bucket()
        self.assertEqual(bucket.select_checkpoint(run_directory=self.run_directory, step=None),
                         'step_000100000.pt')

    def test_explicit_step_is_used(self) -> None:
        bucket = self.make_bucket()
        self.assertEqual(bucket.select_checkpoint(run_directory=self.run_directory, step='step_000014000'),
                         'step_000014000.pt')

    def test_explicit_step_must_exist(self) -> None:
        bucket = self.make_bucket()
        with self.assertRaises(FileNotFoundError):
            bucket.select_checkpoint(run_directory=self.run_directory, step='step_000022000')

    def test_missing_checkpoints_raise(self) -> None:
        with self.assertRaises(FileNotFoundError):
            make_bucket(objects={}).select_checkpoint(
                run_directory='2026-01-01_00-00-00_slug-i000000000000', step=None)


class EnsureLocalTests(unittest.TestCase):
    def make_bucket(self) -> run_suite.CheckpointBucket:
        self.run_directory = '2026-01-01_00-00-00_slug-i000000000000'
        self.checkpoint = 'step_000014000.pt'
        self.key = f'{BUCKET_PREFIX}/{self.run_directory}/{self.checkpoint}'
        self.client = FakeS3Client(objects={self.key: 20})
        return run_suite.CheckpointBucket(client=self.client, bucket='toy-act', prefix=BUCKET_PREFIX)

    def test_downloads_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bucket = self.make_bucket()
            destination = bucket.ensure_local(run_directory=self.run_directory,
                                              checkpoint_name=self.checkpoint, mirror_root=Path(directory))
            self.assertEqual(destination, Path(directory) / self.run_directory / self.checkpoint)
            self.assertEqual(destination.stat().st_size, 20)
            self.assertEqual(self.client.downloads, [self.key])

    def test_reuses_matching_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bucket = self.make_bucket()
            first = bucket.ensure_local(run_directory=self.run_directory,
                                        checkpoint_name=self.checkpoint, mirror_root=Path(directory))
            second = bucket.ensure_local(run_directory=self.run_directory,
                                         checkpoint_name=self.checkpoint, mirror_root=Path(directory))
            self.assertEqual(first, second)
            self.assertEqual(self.client.downloads, [self.key])

    def test_dry_run_skips_the_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bucket = self.make_bucket()
            destination = bucket.ensure_local(run_directory=self.run_directory, checkpoint_name=self.checkpoint,
                                              mirror_root=Path(directory), download=False)
            self.assertFalse(destination.exists())
            self.assertEqual(self.client.downloads, [])


if __name__ == '__main__':
    unittest.main()
