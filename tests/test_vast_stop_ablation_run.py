"""Offline single-run stop checks: one exact run, siblings untouched."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import stop_ablation_run
from workflow_common import Blocked

RUN_A = "2026-10-12_10-36-35_osc_pose_delta_k10-ibeb087d354a2"
RUN_B = "2026-10-12_10-43-56_osc_pose_delta_k100-ibeb087d354a2"


class SelectorTests(unittest.TestCase):
    def test_split_selector_accepts_dashboard_and_bare_forms(self) -> None:
        self.assertEqual(stop_ablation_run.split_selector(selector=f"0000/{RUN_A}"), ("0000", RUN_A))
        self.assertEqual(stop_ablation_run.split_selector(selector=RUN_A), (None, RUN_A))

    def test_split_selector_rejects_non_combo_prefix(self) -> None:
        with self.assertRaises(Blocked):
            stop_ablation_run.split_selector(selector=f"0/{RUN_A}")
        with self.assertRaises(Blocked):
            stop_ablation_run.split_selector(selector=f"0000/{RUN_A}/extra")


class SingleRunStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.combo_a = {"id": "0000", "status": "running", "instance_id": 111,
                        "training_launch_requested": True, "remote_run_name": RUN_A}
        self.combo_b = {"id": "0001", "status": "running", "instance_id": 222,
                        "training_launch_requested": True, "remote_run_name": RUN_B}
        self.iteration = SimpleNamespace(state={"combos": [self.combo_a, self.combo_b]}, journal=Mock())

    def test_combo_prefix_and_run_name_select_the_named_run(self) -> None:
        combo = stop_ablation_run.select_combo(iteration=self.iteration, combo_id="0001", run_name=RUN_B)
        self.assertIs(combo, self.combo_b)

    def test_bare_run_name_selects_the_named_run(self) -> None:
        combo = stop_ablation_run.select_combo(iteration=self.iteration, combo_id=None, run_name=RUN_A)
        self.assertIs(combo, self.combo_a)

    def test_wrong_combo_prefix_or_unknown_run_blocks(self) -> None:
        with self.assertRaises(Blocked):
            stop_ablation_run.select_combo(iteration=self.iteration, combo_id="0001", run_name=RUN_A)
        with self.assertRaises(Blocked):
            stop_ablation_run.select_combo(iteration=self.iteration, combo_id=None, run_name="missing-ixxxxxxxxxxxx")

    def test_stop_cancels_only_the_selected_instance(self) -> None:
        with patch.object(stop_ablation_run, "cancel") as cancel:
            stop_ablation_run.stop(iteration=self.iteration, combo_id="0000", run_name=RUN_A, wait_seconds=0)
        cancel.assert_called_once_with(iteration=self.iteration, instance_id=111, wait_seconds=0)

    def test_stop_without_recorded_instance_blocks(self) -> None:
        self.combo_a.pop("instance_id")
        with patch.object(stop_ablation_run, "cancel") as cancel:
            with self.assertRaises(Blocked):
                stop_ablation_run.stop(iteration=self.iteration, combo_id="0000", run_name=RUN_A, wait_seconds=0)
        cancel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
