# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint boundary saves and Slurm handoff without GPUs or a cluster."""

import importlib.util
import queue
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rlinf.utils.train_budget import TrainBudget, atomic_json

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "dreamdojo_chain", REPO / "docker/dreamdojo/chain.py"
)
CHAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHAIN)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("3.9h", 14040),
        ("7m", 420),
        ("1.5s", 1.5),
        ("04:00:00", 14400),
        ("1-02:03:04", 93784),
    ],
)
def test_duration(value, expected):
    assert CHAIN.duration(value) == expected


@pytest.mark.parametrize("value", ["0", "-1h", "nan", "inf", "3.9hours", ""])
def test_duration_rejects_invalid(value):
    with pytest.raises(ValueError):
        CHAIN.duration(value)


def budget(tmp_path, monkeypatch, **extra):
    monkeypatch.setattr("rlinf.utils.train_budget.time.time", lambda: 1000)
    return TrainBudget(
        {
            "RLINF_TRAIN_DEADLINE": "3000",
            "RLINF_CHAIN_DIR": str(tmp_path),
            "RLINF_CHAIN_JOB_ID": "10",
            "RLINF_CHAIN_STEP_SECONDS": "1000",
            "RLINF_CHAIN_EVAL_SECONDS": "600",
            "RLINF_CHAIN_SAVE_SECONDS": "120",
            **extra,
        }
    )


def test_budget_disabled_for_existing_entrypoints(monkeypatch):
    monkeypatch.delenv("RLINF_TRAIN_DEADLINE", raising=False)
    assert TrainBudget.from_env() is None


def test_reserves_scheduled_and_final_eval_without_changing_interval(
    tmp_path, monkeypatch
):
    control = budget(tmp_path, monkeypatch)
    assert not control.should_stop(4, 1000, 5)
    assert control.should_stop(5, 1000, 5)
    assert control.should_stop(4, 4, 5)
    assert not control.should_stop(5, 1000, -1)
    control.observe("step", 2000)
    assert control.should_stop(4, 1000, 5)
    control.observe("step", 1)
    assert control.estimates["step"] == 2000


def test_stop_file_and_atomic_marker(tmp_path, monkeypatch):
    control = budget(tmp_path / "chain", monkeypatch)
    checkpoint = tmp_path / "run/checkpoints/global_step_7"
    assert not (control.directory / "latest.json").exists()
    control.saved(str(checkpoint), 7, 1000)
    assert CHAIN.read_json(control.directory / "latest.json") == CHAIN.read_json(
        checkpoint / "training_state.json"
    )
    assert not list(tmp_path.rglob("*.tmp"))
    (control.directory / "STOP").touch()
    assert control.should_stop(8, 1000, 5)
    control.finish(7, 1000, True)
    assert CHAIN.read_json(control.directory / "result-10.json")["status"] == "continue"
    control.finish(1000, 1000, False)
    assert CHAIN.read_json(control.directory / "result-10.json")["status"] == "complete"


def completed(directory, outputs, job="10", step=1, max_steps=2):
    relative = f"{job}-train/test/checkpoints/global_step_{step}"
    checkpoint = outputs / relative
    dcp = checkpoint / "actor/dcp_checkpoint"
    dcp.mkdir(parents=True, exist_ok=True)
    for name in [".metadata", *(f"__{rank}_0.distcp" for rank in range(8))]:
        (dcp / name).write_text("test-state")
    marker = {
        "version": 1,
        "checkpoint": f"/outputs/{relative}",
        "global_step": step,
        "max_steps": max_steps,
        "job_id": job,
        "estimates": {"step": 12, "save": 2, "eval": 10},
    }
    atomic_json(checkpoint / "training_state.json", marker)
    atomic_json(directory / "latest.json", marker)
    return marker, checkpoint


def test_latest_ignores_unmarked_directories_and_rejects_partial_save(tmp_path):
    directory, outputs = tmp_path / "chains/10", tmp_path
    assert CHAIN.latest_checkpoint(directory, outputs) is None
    marker, checkpoint = completed(directory, outputs)
    assert CHAIN.latest_checkpoint(directory, outputs) == marker
    (checkpoint / "actor/dcp_checkpoint/__7_0.distcp").unlink()
    with pytest.raises(ValueError, match="missing native DCP"):
        CHAIN.latest_checkpoint(directory, outputs)


def test_corrupt_marker_and_path_escape_rejected(tmp_path):
    directory = tmp_path / "chains/10"
    marker, checkpoint = completed(directory, tmp_path)
    atomic_json(checkpoint / "training_state.json", {**marker, "global_step": 2})
    with pytest.raises(ValueError, match="does not match"):
        CHAIN.latest_checkpoint(directory, tmp_path)
    atomic_json(
        directory / "latest.json", {**marker, "checkpoint": "/outputs/../../elsewhere"}
    )
    with pytest.raises(ValueError):
        CHAIN.latest_checkpoint(directory, tmp_path)


@pytest.fixture
def launch(tmp_path, monkeypatch):
    # Source manifests use real files; all generated outputs are temporary.
    for key, value in {
        "SLURM_JOB_ID": "10",
        "REPO_ROOT": str(REPO),
        "RLINF_OUTPUTS": str(tmp_path),
        "CHAIN_ID": "10",
        "RUN_COUNT": "1",
        "MAX_RUNS": "2",
        "CHAIN_TIMEOUT": "7m",
    }.items():
        monkeypatch.setenv(key, value)
    calls = []

    def command(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "scontrol":
            return "JobId=10 Account=test Partition=batch QOS=normal TimeLimit=00:10:00 RunTime=00:00:03 NumNodes=1 NumTasks=1 JobName=example Features=(null) ReqNodeList=(null) ExcNodeList=(null)"
        assert args[0] == "sbatch"
        return "11;cluster\n"

    monkeypatch.setattr(CHAIN.subprocess, "check_output", command)
    return tmp_path / "chains/10", calls


def test_two_link_handoff_preserves_argv_and_restores_checkpoint(
    launch, monkeypatch, tmp_path
):
    directory, calls = launch
    overrides = [
        "runner.max_epochs=2",
        "env.train.initial_image_mixing_weights=[0.34,0.33,0.33]",
        "runner.logger.experiment_name='case with spaces'",
    ]
    runs = []

    def run(args, env, **kwargs):
        runs.append(args)
        job, step = env["RLINF_CHAIN_JOB_ID"], int(env["RUN_COUNT"])
        completed(directory, tmp_path, job, step)
        atomic_json(
            directory / f"result-{job}.json",
            {
                "status": "continue" if step == 1 else "complete",
                "global_step": step,
                "max_steps": 2,
                "job_id": job,
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(CHAIN.subprocess, "run", run)
    assert CHAIN.main(overrides) == 0
    submit, kwargs = calls[-1]
    assert submit[-3:] == overrides  # Commas/spaces remain single argv entries.
    assert "--dependency=afterok:10" in submit
    assert kwargs["env"]["RUN_COUNT"] == "2"
    assert "SLURM_JOB_ID" not in kwargs["env"]
    monkeypatch.setenv("SLURM_JOB_ID", "11")
    monkeypatch.setenv("RUN_COUNT", "2")
    assert CHAIN.main(overrides) == 0
    assert (
        runs[-1][-1]
        == "runner.resume_dir=/outputs/10-train/test/checkpoints/global_step_1"
    )
    assert sum(args[0] == "sbatch" for args, _ in calls) == 1


@pytest.mark.parametrize("code", [1, 137, 143, -15])
def test_errors_and_cancellation_never_resubmit(launch, monkeypatch, tmp_path, code):
    directory, calls = launch

    def run(*args, **kwargs):
        completed(directory, tmp_path)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(CHAIN.subprocess, "run", run)
    assert CHAIN.main([]) != 0
    assert not any(args[0] == "sbatch" for args, _ in calls)


def test_timeout_can_only_resume_completed_progress(launch, monkeypatch, tmp_path):
    directory, calls = launch
    monkeypatch.setattr(
        CHAIN.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=124)
    )
    with pytest.raises(RuntimeError, match="No completed training progress"):
        CHAIN.main([])
    assert not any(args[0] == "sbatch" for args, _ in calls)


def test_changed_config_is_rejected(launch, monkeypatch, tmp_path):
    monkeypatch.setattr(
        CHAIN.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    assert CHAIN.main(["actor.micro_batch_size=8"]) == 1
    with pytest.raises(
        ValueError, match="arguments, assets, runtime or source changed"
    ):
        CHAIN.main(["actor.micro_batch_size=4"])


@pytest.mark.parametrize(
    "stop,max_runs,expected",
    [(False, "2", "submit"), (True, "2", "stop"), (False, "1", "limit")],
)
def test_timeout_with_checkpoint_obeys_stop_and_chain_limit(
    launch, monkeypatch, tmp_path, stop, max_runs, expected
):
    directory, calls = launch
    monkeypatch.setenv("MAX_RUNS", max_runs)

    def run(*args, **kwargs):
        completed(directory, tmp_path)
        if stop:
            (directory / "STOP").touch()
        return SimpleNamespace(returncode=124)

    monkeypatch.setattr(CHAIN.subprocess, "run", run)
    if expected == "limit":
        with pytest.raises(RuntimeError, match="MAX_RUNS"):
            CHAIN.main([])
    else:
        assert CHAIN.main([]) == 0
    assert any(args[0] == "sbatch" for args, _ in calls) == (expected == "submit")


@pytest.mark.parametrize("max_steps,stopped", [(1000, True), (3, False)])
def test_native_loop_saves_at_boundary_and_preserves_global_step(max_steps, stopped):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(runner={})
    runner.cfg.runner = SimpleNamespace(get=lambda *args: False, val_check_interval=5)
    runner.global_step, runner.max_steps = 2, max_steps
    runner.train_budget = Mock(latest=None)
    runner.train_budget.should_stop.side_effect = [False, True]
    runner.actor, runner.env, runner.rollout = Mock(), Mock(), Mock()
    runner.reward = None
    runner.actor_channel = runner.env_channel = runner.rollout_channel = (
        runner.reward_channel
    ) = None
    runner.logger = Mock()
    runner.weight_sync_interval = 1
    runner.overlap_env_bootstrap = False
    runner.timer = lambda *args, **kwargs: nullcontext()
    runner._should_profile_step = lambda step: False
    for name in (
        "update_rollout_weights",
        "_maybe_eval_and_checkpoint",
        "_log_step_metrics",
        "_finish_run",
        "_save_checkpoint",
    ):
        setattr(runner, name, Mock())
    runner.run()
    assert runner.global_step == 3
    runner.actor.set_global_step.assert_called_once_with(2)
    runner.update_rollout_weights.assert_called_once()
    runner._maybe_eval_and_checkpoint.assert_called_once_with(2)
    runner._save_checkpoint.assert_called_once()
    runner._finish_run.assert_called_once()
    runner.train_budget.finish.assert_called_once_with(3, max_steps, stopped)


def test_save_failure_does_not_publish_native_checkpoint(tmp_path, monkeypatch):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.train_budget = budget(tmp_path / "chain", monkeypatch)
    runner.logger = Mock()
    runner.global_step, runner.max_steps = 1, 2
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            logger=SimpleNamespace(log_path=str(tmp_path), experiment_name="test")
        )
    )
    runner.actor = Mock()
    runner.actor.save_checkpoint.return_value.wait.side_effect = RuntimeError(
        "rank 7 failed"
    )
    with pytest.raises(RuntimeError, match="rank 7 failed"):
        runner._save_checkpoint()
    assert not (tmp_path / "chain/latest.json").exists()
    runner.actor.save_checkpoint.return_value.wait.side_effect = None
    runner._save_checkpoint()
    assert CHAIN.read_json(tmp_path / "chain/latest.json")["global_step"] == 1


def test_logging_exception_does_not_leave_unfinished_queue_item():
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.stop_logging = False
    runner.log_queue = queue.Queue()

    def failing_log():
        runner.stop_logging = True
        raise ValueError("test logging failure")

    runner.log_queue.put((failing_log, ()))
    runner._log_worker()
    assert runner.log_queue.unfinished_tasks == 0


def test_finish_drains_logs_before_stopping_thread():
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.stop_logging = False
    runner.log_thread, runner.metric_logger = Mock(), Mock()

    def drain():
        assert not runner.stop_logging

    runner.log_queue = SimpleNamespace(join=drain)
    runner._finish_run()
    assert runner.stop_logging
    runner.log_thread.join.assert_called_once_with(timeout=1.0)
    runner.metric_logger.finish.assert_called_once()


def test_shell_syntax():
    for script in ("train_chain.slurm", "run_cluster.slurm"):
        subprocess.run(
            ["bash", "-n", str(REPO / "docker/dreamdojo" / script)], check=True
        )
