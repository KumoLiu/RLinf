# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Host-side Slurm handoff for native RLinf training; standard library only."""

import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def duration(value: str) -> float:
    """Parse a positive timeout (s/m/h/d) or Slurm [days-]HH:MM:SS."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", value)
    if match:
        seconds = (
            float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
        )
    else:
        match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
        if not match:
            raise ValueError(f"Invalid duration: {value}")
        days, hours, minutes, seconds = map(int, (part or 0 for part in match.groups()))
        seconds += 60 * minutes + 3600 * hours + 86400 * days
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid duration: {value}")
    return seconds


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def latest_checkpoint(directory: Path, outputs: Path) -> dict | None:
    """Reject partial saves, malformed pointers and paths outside mounted outputs."""
    pointer = directory / "latest.json"
    if not pointer.exists():
        return None
    latest = read_json(pointer)
    relative = Path(latest["checkpoint"]).relative_to("/outputs")
    checkpoint = (outputs / relative).resolve()
    checkpoint.relative_to(outputs.resolve())
    if checkpoint.name != f"global_step_{latest['global_step']}":
        raise ValueError("Checkpoint step/path mismatch")
    if read_json(checkpoint / "training_state.json") != latest:
        raise ValueError("Latest pointer does not match completed checkpoint marker")
    dcp = checkpoint / "actor/dcp_checkpoint"
    if not (dcp / ".metadata").is_file() or len(list(dcp.glob("*.distcp"))) != 8:
        raise ValueError("Completed checkpoint is missing native DCP state")
    if any(path.stat().st_size == 0 for path in dcp.iterdir() if path.is_file()):
        raise ValueError("Empty native DCP checkpoint file")
    return latest


def job_options(job: dict) -> list[str]:
    """Keep the allocation's scheduling choices, not its assigned physical node."""
    options = []
    for field, flag in (
        ("Account", "account"),
        ("Partition", "partition"),
        ("QOS", "qos"),
        ("TimeLimit", "time"),
        ("JobName", "job-name"),
        ("Features", "constraint"),
        ("ReqNodeList", "nodelist"),
        ("ExcNodeList", "exclude"),
        ("Reservation", "reservation"),
    ):
        value = job.get(field)
        if value and value not in ("(null)", "N/A"):
            options.append(f"--{flag}={value}")
    return options


def main(overrides: list[str]) -> int:
    environ = os.environ.copy()
    job_id = environ["SLURM_JOB_ID"]
    chain_id = environ.get("CHAIN_ID", job_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", chain_id):
        raise ValueError("CHAIN_ID must be a simple directory name")
    run_count, max_runs = (
        int(environ.get("RUN_COUNT", 1)),
        int(environ.get("MAX_RUNS", 6)),
    )
    if not 1 <= run_count <= max_runs:
        raise ValueError("Require 1 <= RUN_COUNT <= MAX_RUNS")
    repo = Path(environ["REPO_ROOT"]).resolve()
    base = Path(
        environ.get("CLUSTER_BASE", "/lustre/fsw/portfolios/healthcareeng/users/yunl")
    )
    outputs = Path(environ.get("RLINF_OUTPUTS", base / "outputs/rlinf")).resolve()
    directory = outputs / "chains" / chain_id
    directory.mkdir(parents=True, exist_ok=True)
    # The lock is intentionally NOT inherited by srun/containers/child jobs.
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "STOP").exists():
            print(f"Chain stopped by {directory / 'STOP'}", flush=True)
            return 0
        # Reject changed arguments/code within a chain; files stay outside run logs.
        files = [
            repo / "docker/dreamdojo" / name
            for name in ("chain.py", "train_chain.slurm", "run_cluster.slurm")
        ]
        if environ.get("DREAMDOJO_KIR_SOURCE_ROOT"):
            source = Path(environ["DREAMDOJO_KIR_SOURCE_ROOT"])
            files += [
                source / name
                for name in (
                    "rlinf/data/datasets/lerobot_world_model.py",
                    "rlinf/envs/world_model/world_model_dreamdojo_env.py",
                    "rlinf/envs/world_model/dreamdojo_reward.py",
                    "examples/embodiment/config/env/dreamdojo_trocar.yaml",
                    "toolkits/world_model/dreamdojo_validation.py",
                )
            ]
        if environ.get("RLINF_CHAIN_SOURCE_ROOT"):
            source = Path(environ["RLINF_CHAIN_SOURCE_ROOT"])
            files += [
                source / name
                for name in (
                    "rlinf/runners/embodied_runner.py",
                    "rlinf/utils/train_budget.py",
                )
            ]
        manifest = {
            "overrides": overrides,
            "sources": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
            "environment": {
                key: value
                for key, value in environ.items()
                if key.startswith(
                    (
                        "DREAMDOJO_",
                        "GR00T_",
                        "RLINF_",
                        "NCCL_",
                        "HF_CACHE",
                        "CONTAINER_IMAGE",
                    )
                )
                and not key.startswith(("RLINF_TRAIN_", "RLINF_CHAIN_"))
            },
        }
        manifest_path = directory / "launch.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError(
                    "Chain arguments, assets, runtime or source changed; use a new CHAIN_ID"
                )
        elif run_count != 1:
            raise ValueError("A continuation requires the original launch manifest")
        else:
            write_json(manifest_path, manifest)
        if (directory / f"submitted-{job_id}.json").exists():
            raise ValueError("This job has already submitted a continuation")
        latest = latest_checkpoint(directory, outputs)
        if latest and latest["global_step"] >= latest["max_steps"]:
            print("Chain already reached the requested training length", flush=True)
            return 0
        if run_count > 1 and latest is None:
            raise ValueError("Continuation has no completed checkpoint")
        initial_step = latest["global_step"] if latest else 0
        if latest is None:
            for argument in overrides:
                if (
                    argument.startswith("runner.resume_dir=")
                    and argument != "runner.resume_dir=null"
                ):
                    match = re.search(
                        r"/global_step_(\d+)/?$", argument.split("=", 1)[1]
                    )
                    if not match:
                        raise ValueError(
                            "Initial resume_dir must end in global_step_<N>"
                        )
                    initial_step = int(match[1])
        details = subprocess.check_output(
            ["scontrol", "show", "job", job_id, "-o"], text=True
        )
        job = dict(re.findall(r"(\S+?)=(\S+)", details))
        if job.get("NumNodes") != "1" or job.get("NumTasks") != "1":
            raise ValueError(
                "This chain entry supports one node / one task / eight GPUs only"
            )
        budget = duration(environ.get("CHAIN_TIMEOUT", "3.9h"))
        # RunTime includes batch/container startup; keep the same absolute deadline.
        remaining = budget - (
            duration(job["RunTime"]) if job["RunTime"] != "00:00:00" else 0
        )
        if budget + 90 > duration(job["TimeLimit"]) or remaining <= 0:
            raise ValueError(
                "CHAIN_TIMEOUT must leave at least 90 seconds before Slurm TimeLimit"
            )
        environ.update(
            {
                "CHAIN_ID": chain_id,
                "RUN_COUNT": str(run_count),
                "MAX_RUNS": str(max_runs),
                "RLINF_TRAIN_DEADLINE": str(time.time() + remaining),
                "RLINF_CHAIN_DIR": f"/outputs/chains/{chain_id}",
                "RLINF_CHAIN_JOB_ID": job_id,
            }
        )
        for name, default in (("step", 1200), ("eval", 600), ("save", 120)):
            estimate = max(
                float(environ.get(f"CHAIN_{name.upper()}_SECONDS", default)),
                (latest or {}).get("estimates", {}).get(name, 0),
            )
            environ[f"RLINF_CHAIN_{name.upper()}_SECONDS"] = str(estimate)
        arguments = list(overrides)
        if latest:
            arguments.append(f"runner.resume_dir={latest['checkpoint']}")
        print(
            f"Chain {chain_id} link {run_count}/{max_runs}, resume step {initial_step}, budget remaining {remaining:.0f}s",
            flush=True,
        )
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=60s",
            f"{remaining}s",
            "bash",
            str(repo / "docker/dreamdojo/run_cluster.slurm"),
            "train",
            *arguments,
        ]
        code = subprocess.run(command, env=environ, check=False).returncode
        result_path = directory / f"result-{job_id}.json"
        result = read_json(result_path) if result_path.exists() else {}
        # Never turn ordinary failures/cancellation into retry loops.
        if code not in (0, 124):
            return code if code > 0 else 128 - code
        if code == 0 and result.get("job_id") != job_id:
            raise RuntimeError(
                "Native runner did not publish this job's result (old image/source?)"
            )
        latest = latest_checkpoint(directory, outputs)
        if code == 0 and result.get("status") == "complete":
            if (
                latest is None
                or latest["global_step"] != result["global_step"]
                or result["global_step"] < result["max_steps"]
            ):
                raise RuntimeError("Completed result has no matching final checkpoint")
            print("Native training completed; no continuation", flush=True)
            return 0
        if code == 0 and result.get("status") != "continue":
            raise RuntimeError(
                "Native runner did not publish a handoff (old image/source?)"
            )
        if latest is None or latest["global_step"] <= initial_step:
            raise RuntimeError(
                "No completed training progress; refusing an automatic retry"
            )
        if (directory / "STOP").exists():
            print("STOP requested; checkpoint saved, no continuation", flush=True)
            return 0
        if run_count >= max_runs:
            raise RuntimeError(
                f"MAX_RUNS={max_runs} reached; checkpoint retained at {latest['checkpoint']}"
            )
        environ["RUN_COUNT"] = str(run_count + 1)
        # Do not export stale Slurm task settings to the next independent allocation.
        environ = {
            key: value
            for key, value in environ.items()
            if not key.startswith(("SLURM_", "SBATCH_", "RLINF_TRAIN_", "RLINF_CHAIN_"))
            or key == "RLINF_CHAIN_SOURCE_ROOT"
        }
        submit = [
            "sbatch",
            "--parsable",
            "--export=ALL",
            f"--dependency=afterok:{job_id}",
            *job_options(job),
            f"--chdir={repo}",
            f"--output={directory}/slurm-%j.out",
            f"--error={directory}/slurm-%j.err",
            str(repo / "docker/dreamdojo/train_chain.slurm"),
            *overrides,
        ]
        child = (
            subprocess.check_output(submit, env=environ, text=True)
            .strip()
            .split(";")[0]
        )
        if not child.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {child}")
        write_json(
            directory / f"submitted-{job_id}.json",
            {
                "job_id": job_id,
                "next_job_id": child,
                "global_step": latest["global_step"],
                "exit_code": code,
            },
        )
        print(
            f"Saved step {latest['global_step']}; submitted continuation {child} after successful exit of {job_id}",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
