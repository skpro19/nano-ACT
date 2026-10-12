"""Offline bulk-stop checks: signal every launched run, delete nothing."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HELPERS = Path(__file__).resolve().parents[1] / ".pi/prompts/scripts/vast-train"
sys.path.insert(0, str(HELPERS))
import stop_ablation_runs
from workflow_common import Blocked


def launched(*, combo_id: str, instance_id: int, status: str = "running") -> dict:
    return {"id": combo_id, "status": status, "instance_id": instance_id,
            "training_launch_requested": True}


class StopAblationRunsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.iteration = SimpleNamespace(state={"combos": []}, journal=Mock())

    def stop(self, *, wait_seconds: int = 0) -> int:
        return stop_ablation_runs.stop(iteration=self.iteration, wait_seconds=wait_seconds)

    def test_every_launched_run_is_signaled_before_any_wait(self) -> None:
        self.iteration.state["combos"] = [
            launched(combo_id="0000", instance_id=111),
            launched(combo_id="0001", instance_id=222, status="awaiting_handoff"),
        ]
        order = []
        with patch.object(stop_ablation_runs, "request_cancel",
                          side_effect=lambda **kwargs: order.append(("request", kwargs["instance_id"])) or True), \
                patch.object(stop_ablation_runs, "await_removal",
                             side_effect=lambda **kwargs: order.append(("await", kwargs["instance_id"]))):
            self.assertEqual(self.stop(), 0)
        self.assertEqual(order, [("request", 111), ("request", 222), ("await", 111), ("await", 222)])

    def test_unlaunched_and_terminal_combos_are_untouched(self) -> None:
        self.iteration.state["combos"] = [
            {"id": "0000", "status": "provisioning", "instance_id": 111},
            launched(combo_id="0001", instance_id=222, status="failed"),
            launched(combo_id="0002", instance_id=333, status="abandoned"),
        ]
        with patch.object(stop_ablation_runs, "request_cancel") as request, \
                patch.object(stop_ablation_runs, "await_removal") as await_removal:
            self.assertEqual(self.stop(), 0)
        request.assert_not_called()
        await_removal.assert_not_called()

    def test_no_active_runs_returns_success(self) -> None:
        self.iteration.state["combos"] = [
            launched(combo_id="0000", instance_id=111, status="done")]
        with patch.object(stop_ablation_runs, "request_cancel") as request:
            self.assertEqual(self.stop(), 0)
        request.assert_not_called()

    def test_one_failure_does_not_block_stopping_the_rest(self) -> None:
        self.iteration.state["combos"] = [
            launched(combo_id="0000", instance_id=111),
            launched(combo_id="0001", instance_id=222),
        ]

        def request(**kwargs):
            if kwargs["instance_id"] == 111:
                raise Blocked("watcher is not alive")
            return True

        with patch.object(stop_ablation_runs, "request_cancel", side_effect=request), \
                patch.object(stop_ablation_runs, "await_removal") as await_removal:
            self.assertEqual(self.stop(), 1)
        await_removal.assert_called_once()
        self.assertEqual(await_removal.call_args.kwargs["instance_id"], 222)

    def test_launched_run_without_instance_id_is_reported_not_signaled(self) -> None:
        self.iteration.state["combos"] = [
            {"id": "0000", "status": "running", "training_launch_requested": True}]
        with patch.object(stop_ablation_runs, "request_cancel") as request:
            self.assertEqual(self.stop(), 1)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
