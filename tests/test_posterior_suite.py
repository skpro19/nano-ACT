"""Suite-driver helpers for the posterior-collapse tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'utils' / 'posterior-collpase'))

import run_suite


class ParseTestsTests(unittest.TestCase):
    def test_default_is_every_test(self) -> None:
        self.assertEqual(run_suite.parse_tests(values=[]), [1, 2, 3, 4])

    def test_subset_is_sorted_and_deduplicated(self) -> None:
        self.assertEqual(run_suite.parse_tests(values=['2', '1', '2']), [1, 2])

    def test_unknown_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_suite.parse_tests(values=['9'])


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
    def make_roots(self, *, directory: Path) -> tuple[Path, Path]:
        iteration_root = directory / 'ablations'
        checks_root = directory / 'checkpoints'
        iteration = iteration_root / 'sweep-a' / '20260101T000000Z-aa0000000000'
        iteration.mkdir(parents=True)
        (iteration / 'manifest.json').write_text(json.dumps({'combos': [
            {'id': '0000', 'name': 'slug-iaa0000000000'},
            {'id': '0001', 'name': 'other-iaa0000000000'}]}))
        (checks_root / '2026-01-01_00-00-00_slug-iaa0000000000').mkdir(parents=True)
        return iteration_root, checks_root

    def test_combo_id_selects_the_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, checks_root = self.make_roots(directory=Path(temporary))
            iteration, combo, run_directory = run_suite.resolve_run_scope(
                value='0000/2026-01-01_00-00-00_slug-iaa0000000000',
                iteration_root=iteration_root, checks_root=checks_root)
            self.assertEqual(iteration, 'iaa0000000000')
            self.assertEqual(combo['id'], '0000')
            self.assertEqual(run_directory.name, '2026-01-01_00-00-00_slug-iaa0000000000')

    def test_bare_run_name_matches_by_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, checks_root = self.make_roots(directory=Path(temporary))
            _, combo, run_directory = run_suite.resolve_run_scope(
                value='2026-01-01_00-00-00_slug-iaa0000000000',
                iteration_root=iteration_root, checks_root=checks_root)
            self.assertEqual(combo['id'], '0000')
            self.assertEqual(run_directory.name, '2026-01-01_00-00-00_slug-iaa0000000000')

    def test_unknown_combo_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, checks_root = self.make_roots(directory=Path(temporary))
            with self.assertRaises(ValueError):
                run_suite.resolve_run_scope(
                    value='0002/2026-01-01_00-00-00_slug-iaa0000000000',
                    iteration_root=iteration_root, checks_root=checks_root)

    def test_unknown_iteration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, checks_root = self.make_roots(directory=Path(temporary))
            with self.assertRaises(ValueError):
                run_suite.resolve_run_scope(
                    value='0000/2026-01-01_00-00-00_slug-ibb1111111111',
                    iteration_root=iteration_root, checks_root=checks_root)

    def test_missing_checkpoint_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            iteration_root, checks_root = self.make_roots(directory=Path(temporary))
            with self.assertRaises(FileNotFoundError):
                run_suite.resolve_run_scope(
                    value='0000/2026-01-01_00-00-00_absent-iaa0000000000',
                    iteration_root=iteration_root, checks_root=checks_root)


class CheckpointSelectionTests(unittest.TestCase):
    def test_newest_run_directory_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '2026-01-01_00-00-00_slug-i000000000000').mkdir()
            (root / '2026-02-01_00-00-00_slug-i000000000000').mkdir()
            (root / 'unrelated').mkdir()
            self.assertEqual(run_suite.find_run_directory(run_name='slug-i000000000000', root=root).name,
                             '2026-02-01_00-00-00_slug-i000000000000')

    def test_missing_run_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_suite.find_run_directory(run_name='slug-i000000000000', root=Path(directory))

    def test_latest_checkpoint_is_the_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in ('step_000008000', 'step_000014000', 'step_000100000'):
                (root / f'{step}.pt').touch()
            self.assertEqual(run_suite.select_checkpoint(directory=root, step=None).name, 'step_000100000.pt')

    def test_explicit_step_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'step_000014000.pt').touch()
            self.assertEqual(run_suite.select_checkpoint(directory=root, step='step_000014000').name,
                             'step_000014000.pt')

    def test_explicit_step_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_suite.select_checkpoint(directory=Path(directory), step='step_000100000')


if __name__ == '__main__':
    unittest.main()
