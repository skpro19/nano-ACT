"""Stored-statistic plotting and suite plot-command regression tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / '.pi/prompts/scripts/test-posterior-collapse'
sys.path.insert(0, str(SCRIPTS))

import plot_reconstruction
import plot_test4
import run_suite


def paired_result() -> dict:
    return {'targets': 8, 'demonstrations': 4, 'paired': {
        'prior_sample': {'own_l1': .4, 'absolute': -.1, 'relative': -.25,
                         'absolute_95ci': [-.14, -.08], 'relative_95ci': [-.35, -.2],
                         'classification': 'inconclusive/mixed'}}}


class PlotTests(unittest.TestCase):
    def test_reconstruction_uses_paired_own_and_exact_stored_intervals(self) -> None:
        values = plot_reconstruction.reconstruction_values(result=paired_result())
        np.testing.assert_allclose(values['own'], [.4])
        np.testing.assert_allclose(values['intervention'], [.3])
        np.testing.assert_allclose(values['absolute_ci'], [[-.14, -.08]])
        np.testing.assert_allclose(values['relative_ci'], [[-.35, -.2]])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'results.json'
            source.write_text(json.dumps(paired_result()))
            with patch.object(plot_reconstruction, 'interval_points', wraps=plot_reconstruction.interval_points) as draw:
                plot_reconstruction.plot_result(result=paired_result(), title='Synthetic',
                    output=Path(directory) / 'plot.png', source=source, code=SCRIPTS / 'plot_test2.py')
            np.testing.assert_allclose(draw.call_args_list[0].kwargs['intervals'], [[-.14, -.08]])
            np.testing.assert_allclose(draw.call_args_list[1].kwargs['intervals'], [[-35, -20]])
            self.assertTrue((Path(directory) / 'plot.provenance.json').exists())

    def test_test2_and_test3_cli_agg_smoke_preserves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'k10' / 'step_000100000'
            for test in (2, 3):
                root = checkpoint / f'test-{test}'
                root.mkdir(parents=True)
                source = root / 'results.json'
                source.write_text(json.dumps({'train': {'combined': paired_result(),
                                                         'by_seed': {'42': paired_result()}}}))
                original = source.read_bytes()
                subprocess.run([sys.executable, str(SCRIPTS / f'plot_test{test}.py'),
                                '--checkpoint-dir', str(checkpoint), '--per-seed'], check=True,
                               capture_output=True)
                for output in (root / 'train/01_paired_reconstruction_combined.png',
                               root / 'train/seed-42/01_paired_reconstruction.png'):
                    self.assertTrue(output.read_bytes().startswith(b'\x89PNG'))
                    self.assertTrue(output.with_suffix('.provenance.json').exists())
                self.assertEqual(source.read_bytes(), original)
        self.assertEqual(plot_reconstruction.matplotlib.get_backend().lower(), 'agg')

    def test_test4_combined_uses_root_ci(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'test-4'
            root.mkdir()
            items = {key: {'mean': .01, 'ci95': [.008, .015], 'status': 'meaningful'}
                     for key in plot_test4.KEYS}
            (root / 'results.json').write_text(json.dumps({'train': {
                'combined': {'targets': 8, 'demonstrations': 4, 'metrics': items}}}))
            with patch.object(plot_test4, 'interval_points', wraps=plot_test4.interval_points) as draw:
                plot_test4.plot_combined(root=root, checkpoint_dir=root.parent)
            self.assertEqual(draw.call_args.kwargs['intervals'], [[.008, .015]] * 3)
            self.assertTrue((root / 'train/02_decoder_sensitivity_combined.png').exists())

    def test_suite_dispatches_checkpoint_plotters_for_every_test(self) -> None:
        for test in (1, 2, 3, 4):
            self.assertEqual(run_suite.plot_command(test=test, output=Path('output/checkpoint')),
                             ['uv', 'run', f'.pi/prompts/scripts/test-posterior-collapse/plot_test{test}.py',
                              '--checkpoint-dir', 'output/checkpoint'])


if __name__ == '__main__':
    unittest.main()
