"""Layout rules for the posterior-collapse audit output tree."""

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'utils' / 'posterior-collpase'))

import layout


class VaryingKeysTests(unittest.TestCase):
    def test_single_value_axis_is_not_swept(self) -> None:
        grid = {'action_mode': ['osc_pose_delta'], 'action_chunk_size': [10, 100]}
        self.assertEqual(layout.varying_keys(grid=grid), ['action_chunk_size'])

    def test_repeated_list_values_are_not_swept(self) -> None:
        grid = {'image_keys': [['a', 'b'], ['a', 'b']]}
        self.assertEqual(layout.varying_keys(grid=grid), [])


class ConfigTagTests(unittest.TestCase):
    def test_orders_by_run_name_fields_and_joins_with_plus(self) -> None:
        config = {'action_chunk_size': 50, 'beta': 0.1, 'lr': 1e-5}
        self.assertEqual(layout.config_tag(config=config, varying=['beta', 'action_chunk_size']), 'k50+beta0.1')

    def test_omits_joint_action_mode(self) -> None:
        config = {'action_mode': 'joint_position_aperture', 'action_chunk_size': 10}
        self.assertEqual(layout.config_tag(config=config, varying=['action_mode', 'action_chunk_size']), 'k10')

    def test_renders_non_default_action_mode_first(self) -> None:
        config = {'action_mode': 'osc_pose_delta', 'action_chunk_size': 10}
        self.assertEqual(layout.config_tag(config=config, varying=['action_mode', 'action_chunk_size']),
                         'osc_pose_delta+k10')

    def test_renders_use_z_as_binary_flag(self) -> None:
        config = {'use_z': True, 'action_chunk_size': 100}
        self.assertEqual(layout.config_tag(config=config, varying=['use_z', 'action_chunk_size']), 'k100+z1')


class DirectoryTests(unittest.TestCase):
    def test_tag_adds_a_level_only_when_nonempty(self) -> None:
        base = Path('/tmp/root')
        self.assertEqual(layout.checkpoint_directory(output_root=base, iteration='iabc',
                                                     checkpoint_stem='step_1', tag=''),
                         base / 'iabc' / 'step_1')
        self.assertEqual(layout.checkpoint_directory(output_root=base, iteration='iabc',
                                                     checkpoint_stem='step_1', tag='k10'),
                         base / 'iabc' / 'k10' / 'step_1')

    def test_assert_checkpoint_rejects_a_different_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'checkpoint.sha256').write_text('a' * 64 + '  step.pt\n')
            layout.assert_checkpoint(directory=Path(directory), checkpoint_sha='a' * 64)
            with self.assertRaises(RuntimeError):
                layout.assert_checkpoint(directory=Path(directory), checkpoint_sha='b' * 64)


class ResolveTagTests(unittest.TestCase):
    def test_explicit_tag_wins(self) -> None:
        config = {'action_chunk_size': 50, 'beta': 0.1}
        self.assertEqual(layout.resolve_tag(iteration='i000000000000', config=config, tag='custom'), 'custom')

    def test_sweep_spec_drives_the_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / 'sweep.toml'
            spec.write_text('[grid]\naction_chunk_size = [50, 100]\nbeta = [0.1, 10]\n')
            config = {'action_chunk_size': 100, 'beta': 10}
            self.assertEqual(layout.resolve_tag(iteration='i000000000000', config=config, sweep=spec), 'k100+beta10')

    def test_single_combination_sweep_has_no_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / 'sweep.toml'
            spec.write_text('[grid]\naction_chunk_size = [100]\nbeta = [10]\n')
            self.assertEqual(layout.resolve_tag(iteration='i000000000000', config={'action_chunk_size': 100, 'beta': 10},
                                                sweep=spec), '')


if __name__ == '__main__':
    unittest.main()
