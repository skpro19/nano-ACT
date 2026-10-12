"""Suite-driver helpers for the posterior-collapse tests."""

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
