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

import traceback
from multiprocessing.connection import Connection
from queue import Empty

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()
    isaac_env = sim_app = None
    try:
        created = env_fn_wrapper.x()
        # Kit factories return (env, app); native backends return only env.
        isaac_env, sim_app = created if isinstance(created, tuple) else (created, None)
        device = isaac_env.device
        while True:
            cmd = child_remote.recv()
            if cmd == "reset":
                reset_index, reset_seed, options = reset_idx_queue.get()
                kwargs = {"seed": reset_seed}
                if reset_index is not None:
                    kwargs["env_ids"] = reset_index.to(device)
                if options is not None:
                    kwargs["options"] = options
                reset_result = isaac_env.reset(**kwargs)
                obs_queue.put(reset_result)
            elif cmd == "step":
                input_action = action_queue.get()
                step_result = isaac_env.step(input_action)
                obs_queue.put(step_result)
            elif cmd == "close":
                break
            elif cmd == "device":
                obs_queue.put(device)
            else:
                raise NotImplementedError(f"Unknown environment command: {cmd}")
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        obs_queue.put(
            RuntimeError(
                f"IsaacLab environment subprocess failed:\n{traceback.format_exc()}"
            )
        )
    finally:
        try:
            if isaac_env is not None:
                isaac_env.close()
        finally:
            try:
                if sim_app is not None:
                    sim_app.close()
            finally:
                child_remote.close()


class SubProcIsaacLabEnv:
    """Run an (env, app) factory or an app-free native environment in a subprocess."""

    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.obs_queue = ctx.Queue()
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.reset_idx,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()
        self.child_remote.close()

    def _receive(self):
        """Surface child errors/exits without imposing an inference deadline."""
        while True:
            try:
                result = self.obs_queue.get(timeout=1)
            except Empty:
                if self.isaac_lab_process.is_alive():
                    continue
                self.close()
                raise RuntimeError(
                    f"IsaacLab environment subprocess exited: {self.isaac_lab_process.exitcode}"
                ) from None
            if isinstance(result, Exception):
                self.close()
                raise result
            return result

    def reset(self, seed=None, env_ids=None, *, options=None):
        """Forward task reset options without changing legacy reset calls."""
        if self.parent_remote.closed:
            raise RuntimeError("IsaacLab environment subprocess is closed")
        self.parent_remote.send("reset")
        self.reset_idx.put((env_ids, seed, options))
        return self._receive()

    def step(self, action: torch.Tensor):
        """Forward batched actions or action chunks without reshaping."""
        if self.parent_remote.closed:
            raise RuntimeError("IsaacLab environment subprocess is closed")
        self.parent_remote.send("step")
        self.action_queue.put(action)
        return self._receive()

    def close(self):
        """Release the child and queues; repeated calls are safe."""
        if self.parent_remote.closed:
            return
        try:
            self.parent_remote.send("close")
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.isaac_lab_process.join(timeout=5)
        if self.isaac_lab_process.is_alive():
            self.isaac_lab_process.terminate()
            self.isaac_lab_process.join(timeout=5)
        if self.isaac_lab_process.is_alive():
            self.isaac_lab_process.kill()
            self.isaac_lab_process.join()
        self.parent_remote.close()
        self.child_remote.close()
        for queue in (self.action_queue, self.obs_queue, self.reset_idx):
            queue.close()
            queue.cancel_join_thread()

    def device(self):
        """Return the child environment's device."""
        if self.parent_remote.closed:
            raise RuntimeError("IsaacLab environment subprocess is closed")
        self.parent_remote.send("device")
        return self._receive()
