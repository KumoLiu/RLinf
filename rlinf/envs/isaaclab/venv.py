# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from queue import Empty

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _torch_worker(child_remote, parent_remote, env_fn_wrapper, obs_queue, log_path):
    parent_remote.close()
    isaac_env = sim_app = None
    with ExitStack() as stack:
        try:
            if log_path is not None:
                stream = stack.enter_context(open(log_path, "x", buffering=1))
                stack.enter_context(redirect_stdout(stream))
                stack.enter_context(redirect_stderr(stream))
            created = env_fn_wrapper.x()
            # PhysX factories return (env, SimulationApp); headless backends need no app.
            isaac_env, sim_app = (
                created if isinstance(created, tuple) else (created, None)
            )
            obs_queue.put((True, isaac_env.device))
            while True:
                cmd, payload = child_remote.recv()
                if cmd == "close":
                    break
                if cmd == "reset":
                    if "env_ids" in payload:
                        payload["env_ids"] = payload["env_ids"].to(isaac_env.device)
                    result = isaac_env.reset(**payload)
                elif cmd == "step":
                    result = isaac_env.step(payload)
                else:
                    raise ValueError(f"Unknown environment command: {cmd}")
                obs_queue.put((True, result))
        except EOFError:
            pass
        except BaseException:
            obs_queue.put((False, traceback.format_exc()))
        finally:
            try:
                if isaac_env is not None:
                    isaac_env.close()
            finally:
                if sim_app is not None:
                    sim_app.close()
                child_remote.close()


class SubProcIsaacLabEnv:
    """Run a native environment in a spawned process, with optional Kit ownership.

    Existing factories returning ``(env, app)`` remain supported. A factory may
    instead return an environment alone. ``timeout_s=None`` retains unbounded
    inference waits, but child exceptions and exits are always reported.
    """

    def __init__(
        self,
        env_fn,
        *,
        timeout_s: float | None = None,
        log_path: str | Path | None = None,
    ):
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = timeout_s
        self._closed = False
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.obs_queue = ctx.Queue()
        if log_path is not None:
            log_path = str(Path(log_path).absolute())
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.obs_queue,
            log_path,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        try:
            self.isaac_lab_process.start()
            self.child_remote.close()
            self._device = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        deadline = None if self.timeout_s is None else time.monotonic() + self.timeout_s
        while True:
            remaining = 1.0 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("IsaacLab environment subprocess timed out")
            try:
                ok, result = self.obs_queue.get(timeout=min(remaining, 1.0))
            except Empty:
                if not self.isaac_lab_process.is_alive():
                    raise RuntimeError(
                        f"IsaacLab environment subprocess exited: {self.isaac_lab_process.exitcode}"
                    ) from None
                continue
            if not ok:
                raise RuntimeError(f"IsaacLab environment subprocess failed:\n{result}")
            return result

    def _request(self, command, payload):
        if self._closed:
            raise RuntimeError("IsaacLab environment subprocess is closed")
        try:
            self.parent_remote.send((command, payload))
            return self._receive()
        except BaseException:
            self.close()
            raise

    def reset(self, seed=None, env_ids=None, *, options=None):
        """Forward reset arguments, preserving legacy optional environment IDs."""
        kwargs = {"seed": seed}
        if env_ids is not None:
            kwargs["env_ids"] = env_ids
        if options is not None:
            kwargs["options"] = options
        return self._request("reset", kwargs)

    def step(self, action: torch.Tensor):
        """Forward native actions or action chunks without changing their shape."""
        return self._request("step", action)

    def close(self):
        """Close the child and its resources; safe after partial startup or errors."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.isaac_lab_process.pid is not None:
                try:
                    self.parent_remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.isaac_lab_process.join(timeout=5)
                if self.isaac_lab_process.is_alive():
                    self.isaac_lab_process.terminate()
                    self.isaac_lab_process.join(timeout=5)
                if self.isaac_lab_process.is_alive():
                    self.isaac_lab_process.kill()
                    self.isaac_lab_process.join()
        finally:
            self.parent_remote.close()
            self.child_remote.close()
            self.obs_queue.close()
            self.obs_queue.cancel_join_thread()

    def device(self):
        return self._device
