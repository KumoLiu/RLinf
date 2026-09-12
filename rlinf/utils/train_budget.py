# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in wall-clock budget for synchronous embodied training.

This controls *when* the native runner saves; it does not replace its FSDP
checkpoint format. Only a completed collective save may publish a latest pointer.
"""

import json
import math
import os
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    """Publish small metadata atomically on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class TrainBudget:
    """Stop at iteration boundaries, reserving time for eval and checkpoint IO."""

    def __init__(self, environ: dict):
        self.deadline = float(environ["RLINF_TRAIN_DEADLINE"])
        self.directory = Path(environ["RLINF_CHAIN_DIR"])
        self.job_id = environ["RLINF_CHAIN_JOB_ID"]
        self.estimates = {
            name: float(environ.get(f"RLINF_CHAIN_{name.upper()}_SECONDS", default))
            for name, default in (("step", 1200), ("eval", 600), ("save", 120))
        }
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.deadline, *self.estimates.values())
        ):
            raise ValueError("Training deadline and budget estimates must be positive")
        self.latest = None

    @classmethod
    def from_env(cls):
        """Leave ordinary local/train/eval jobs unchanged unless explicitly enabled."""
        if not os.environ.get("RLINF_TRAIN_DEADLINE"):
            return None
        return cls(os.environ)

    def observe(self, name: str, seconds: float) -> None:
        """Keep conservative estimates; never shrink below the configured floor."""
        self.estimates[name] = max(self.estimates[name], seconds)

    def should_stop(self, next_step: int, max_steps: int, val_interval: int) -> bool:
        """Include scheduled native eval instead of silently skipping validation."""
        if (self.directory / "STOP").exists():
            return True
        will_eval = val_interval > 0 and (
            next_step % val_interval == 0 or next_step == max_steps
        )
        needed = (
            1.2 * (self.estimates["step"] + will_eval * self.estimates["eval"])
            + self.estimates["save"]
        )
        return time.time() + needed >= self.deadline

    def saved(self, checkpoint: str, step: int, max_steps: int) -> None:
        """Call ONLY after all ranks have successfully saved native training state."""
        self.latest = {
            "version": 1,
            "checkpoint": checkpoint,
            "global_step": step,
            "max_steps": max_steps,
            "job_id": self.job_id,
            "estimates": self.estimates.copy(),
        }
        atomic_json(Path(checkpoint) / "training_state.json", self.latest)
        atomic_json(self.directory / "latest.json", self.latest)

    def finish(self, step: int, max_steps: int, stopped: bool) -> None:
        """Publish a handoff only after the runner has flushed its logs."""
        atomic_json(
            self.directory / f"result-{self.job_id}.json",
            {
                "status": "continue" if stopped and step < max_steps else "complete",
                "global_step": step,
                "max_steps": max_steps,
                "job_id": self.job_id,
            },
        )
