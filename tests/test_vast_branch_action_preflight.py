"""Offline branch identity and shared local/remote dataset validation contracts."""
import argparse
import builtins
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import h5py
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_action_modes import make_dataset
from test_vast_workflow import COMMIT, HELPERS, LocalFixture
import iteration
import provision
import setup_run
import workflow
import workflow_common as common
from scripts import action_modes, train_v2


class BranchTests(LocalFixture):
    def test_new_identity_and_legacy_compatibility(self) -> None:
        saved = self.make_iteration(branch="feat/actions")
        self.assertEqual(saved.branch, "feat/actions")
        self.assertEqual(saved.state["git_branch"], saved.branch)
        saved.state["git_branch"] = "other"
        with self.assertRaises(common.Blocked):
            saved.validate()
        saved.manifest.pop("git_branch")
        saved.state.pop("git_branch")
        saved.state.pop("git_commit")
        common.write_json(path=saved.directory / "manifest.json", value=saved.manifest)
        saved.state["manifest_sha256"] = common.digest(saved.directory / "manifest.json")
        common.write_json(path=saved.directory / "state.json", value=saved.state)
        before = (saved.directory / "manifest.json").read_bytes()
        legacy = iteration.Iteration(directory=saved.directory)
        self.assertEqual(legacy.branch, "act-v2")
        legacy.save()
        self.assertEqual(before, (saved.directory / "manifest.json").read_bytes())
        self.assertNotIn("git_branch", legacy.state)

    def test_saved_invalid_branch_and_commit_identity_fail(self) -> None:
        saved = self.make_iteration()
        saved.state["git_commit"] = "b" * 40
        with self.assertRaises(common.Blocked):
            saved.validate()
        saved.state["git_commit"] = COMMIT
        saved.manifest["git_branch"] = "refs/heads/main"
        common.write_json(path=saved.directory / "manifest.json", value=saved.manifest)
        saved.state["manifest_sha256"] = common.digest(saved.directory / "manifest.json")
        with self.assertRaises(common.Blocked):
            saved.validate()

    def test_resume_inherits_and_conflicts_before_execute(self) -> None:
        saved = self.make_iteration(branch="feat/actions")
        for branch, result in ((None, 0), ("feat/actions", 0), ("act-v2", 1)):
            args = argparse.Namespace(spec=None, config=None, resume=saved.directory.name,
                                      expected_commit=None, branch=branch, plan=False)
            with patch.object(workflow, "parse_args", return_value=args), \
                    patch.object(workflow, "locate", return_value=saved.directory), \
                    patch.object(workflow, "execute") as execute, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(workflow.main(), result)
                self.assertEqual(execute.call_count, int(result == 0))

    def test_all_execution_gates_use_saved_branch(self) -> None:
        saved = self.make_iteration(branch="feat/actions")
        combo = saved.state["combos"][0]
        def provision_combo(**kwargs):
            self.assertEqual(kwargs["branch"], saved.branch)
            combo["status"] = "setup"
        def setup_combo(**kwargs):
            self.assertEqual(kwargs["branch"], saved.branch)
            combo["status"] = "ready_to_launch"
        with patch.object(workflow, "git_preflight", return_value=COMMIT) as gate, \
                patch.object(workflow, "local_preflight"), patch.object(workflow, "ensure_dashboard"), \
                patch.object(workflow, "validate_dataset_source"), \
                patch.object(workflow, "provision", side_effect=provision_combo), \
                patch.object(workflow, "setup", side_effect=setup_combo), \
                patch.object(workflow, "launch"), patch.object(workflow, "handoff", return_value=0):
            workflow.execute(saved)
        self.assertEqual(gate.call_count, 4)
        for call in gate.call_args_list:
            self.assertEqual(call.kwargs["branch"], saved.branch)
            self.assertEqual(call.kwargs["expected"], COMMIT)

    def test_precreate_gate_blocks_before_create(self) -> None:
        for count in range(3):
            attempts = [{"status": "removed", "offer": {"machine_id": index}} for index in range(count)]
            combo = {"id": "0000", "status": "provisioning" if count else "pending", "attempts": attempts}
            with patch.object(provision, "search_offers", return_value=[{"machine_id": 10}]), \
                    patch.object(provision, "git_preflight", side_effect=common.Blocked("gate")) as gate, \
                    patch.object(self.journal, "run") as run:
                with self.assertRaises(common.Blocked):
                    provision.provision(journal=self.journal, combo=combo, directory=self.directory,
                                        commit=COMMIT, branch="feat/actions", save=lambda: None)
            gate.assert_called_once_with(journal=self.journal, expected=COMMIT, branch="feat/actions")
            run.assert_not_called()

    def test_setup_quotes_branch_and_records_env(self) -> None:
        config = self.root / "config.toml"
        config.write_text('name="test"\ndataset="datasets/test.hdf5"\ncheckpoint_every=5\nsteps=10\n')
        run_dir = self.root / "run"
        run_dir.mkdir()
        run = dict(run_dir=str(run_dir), instance_id=1, label="label", host="host", port=22,
                   known_hosts="hosts", offer=dict(id=2, machine_id=3, dph_total=.5), actual_price=.5)
        combo = dict(run=run)
        lease = dict(TB_PORT=6006)
        with patch.object(self.journal, "run") as command, \
                patch.object(setup_run, "transfer_credentials"), patch.object(setup_run, "prepare_dataset"), \
                patch.object(setup_run, "allocate_lease", return_value=lease):
            setup_run.setup(journal=self.journal, combo=combo, config_path=config,
                            commit=COMMIT, branch="feat/actions", save=lambda: None)
        script = command.call_args_list[0].kwargs["input_text"]
        self.assertIn("export GIT_BRANCH=feat/actions\n", script)
        self.assertIn('git clone --branch "$GIT_BRANCH" --single-branch', script)
        self.assertIn('rev-parse HEAD)', script)
        self.assertIn("GIT_BRANCH=feat/actions", (run_dir / "setup.env").read_text())


@pytest.mark.parametrize("branch", ["-evil", "refs/heads/main", "main~1", "a..b", "a//b", "a/.b", "a.lock", "a/", "a.", "a;echo", "HEAD", "", None])
def test_invalid_branches(branch) -> None:
    with pytest.raises(common.Blocked):
        common.validate_branch(branch)


def test_git_gate_default_and_custom(tmp_path) -> None:
    journal = Mock(directory=tmp_path)
    journal.run.return_value = subprocess.CompletedProcess([], 0, COMMIT + "\n", "")
    for branch in ("act-v2", "feat/actions"):
        assert common.git_preflight(journal=journal, branch=branch) == COMMIT
        assert journal.run.call_args.kwargs["args"][2:4] == ["--branch", branch]


def test_cli_branch_default_and_explicit() -> None:
    for extra, expected in (([], None), (["--branch", "feat/actions"], "feat/actions")):
        with patch.object(sys, "argv", ["workflow", "--config", "config.toml", *extra]):
            assert workflow.parse_args().branch == expected


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("mode", [None, action_modes.JOINT, action_modes.OSC])
@pytest.mark.parametrize("defect", [None, "actions", "metadata", "images", "nonfinite"])
def test_dataset_preflight(tmp_path, monkeypatch, remote, mode, defect) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "datasets/test.hdf5"
    path.parent.mkdir()
    make_dataset(path=path)
    with h5py.File(path, "r+") as file:
        # Break the second demo to prove validation does not stop at the first.
        if defect == "actions":
            del file["data/demo_1/actions"]
        elif defect == "metadata":
            file["data"].attrs["source_env_args"] = "not json"
        elif defect == "images":
            del file["data/demo_1/obs/agentview_image"]
        elif defect == "nonfinite":
            file["data/demo_1/obs/robot0_joint_pos"][0, 0] = float("nan")
    config = {"dataset": "datasets/test.hdf5", "image_keys": ["agentview_image"]}
    if mode is not None:
        config["action_mode"] = mode
    def execute_code(**kwargs):
        code = kwargs["args"][kwargs["args"].index("-c") + 1]
        with patch.object(sys, "argv", ["preflight", "config.toml"]):
            exec(compile(code, "local-preflight", "exec"), {})
        return subprocess.CompletedProcess([], 0, "", "")
    journal = Mock(directory=tmp_path)
    journal.run.side_effect = execute_code
    should_fail = defect in {"images", "nonfinite"} or (mode == action_modes.OSC and defect in {"actions", "metadata"})
    with patch.object(train_v2, "load_config", return_value=config), \
            patch.object(setup_run, "REPO", tmp_path), \
            patch.object(setup_run, "dataset_head", side_effect=AssertionError("No AWS for local files")):
        if remote:
            monkeypatch.setenv("EXPECTED_BYTES", str(path.stat().st_size))
            code = (HELPERS / "verify-dataset.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            run = lambda: exec(compile(code, "remote-preflight", "exec"), {})
        else:
            run = lambda: setup_run.validate_dataset_source(journal=journal, config=config, config_path=Path("config.toml"))
        if should_fail:
            with pytest.raises(ValueError):
                run()
        else:
            run()


def test_s3_only_source_keeps_existence_check(tmp_path) -> None:
    journal = Mock(directory=tmp_path)
    config = dict(dataset="datasets/absent.hdf5", image_keys=["agentview_image"])
    with patch.object(setup_run, "REPO", tmp_path), \
            patch.object(setup_run, "dataset_head", return_value={"ContentLength": 42}) as head:
        setup_run.validate_dataset_source(journal=journal, config=config, config_path=tmp_path / "config.toml")
        head.assert_called_once()
        assert journal.run.call_args.kwargs["timeout"] == 1800


@pytest.mark.parametrize("remote", [False, True])
def test_missing_pinned_helper_fails_clearly(tmp_path, monkeypatch, remote) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "datasets/test.hdf5"
    path.parent.mkdir()
    make_dataset(path=path)
    config = dict(dataset="datasets/test.hdf5", image_keys=["agentview_image"])
    original_import = builtins.__import__
    def import_without_helper(name, *args, **kwargs):
        if name == "scripts.action_modes":
            raise ImportError("historical revision")
        return original_import(name, *args, **kwargs)
    def execute_code(**kwargs):
        code = kwargs["args"][kwargs["args"].index("-c") + 1]
        with patch.object(sys, "argv", ["preflight", "config.toml"]):
            exec(compile(code, "local-preflight", "exec"), {})
    journal = Mock(directory=tmp_path)
    journal.run.side_effect = execute_code
    with patch.object(train_v2, "load_config", return_value=config), \
            patch.object(setup_run, "REPO", tmp_path), \
            patch.object(builtins, "__import__", side_effect=import_without_helper):
        with pytest.raises(RuntimeError, match="Pinned code lacks.*preflight_dataset"):
            if remote:
                code = (HELPERS / "verify-dataset.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
                exec(compile(code, "remote-preflight", "exec"), {})
            else:
                setup_run.validate_dataset_source(journal=journal, config=config, config_path=Path("config.toml"))
