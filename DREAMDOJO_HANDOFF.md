# DreamDojo × GR00T-N1.7 × pick-trocar — Work In Progress Handoff

Snapshot of the RLinf side of the pipeline that mirrors `wan.rst`'s
world-model-driven GRPO, but with **DreamDojo** (Cosmos-Predict2.5) as the
world model and **GR00T N1.7 3B** as the policy trained on the Unitree G1 +
Dex3 trocar-handover teleop data.

Use this file to resume on another machine.

---

## 0. Goal

Reproduce the Wan-GRPO recipe from
`docs/source-en/rst_source/examples/embodied/wan.rst` for a new stack:

| Slot                | Wan reference                                        | Our target                                                                 |
|---------------------|------------------------------------------------------|----------------------------------------------------------------------------|
| Policy              | OpenVLA-OFT (7-D LIBERO, LoRA)                       | GR00T **N1.7** 3B, `EmbodimentTag.NEW_EMBODIMENT`, 28-D dual-arm + Dex3     |
| World model         | Wan2.2 (`RLinf/diffsynth-studio`)                    | **DreamDojo** (Cosmos-Predict2.5, `NVIDIA/DreamDojo`, to be dropped in)     |
| Reward model        | ResNet success classifier, `resnet_rm.pth`           | **4-class progress classifier** (training on another machine)               |
| Init dataset        | `dataset/traj*.npy` + `traj*_kir.npy` (flat dicts)   | **LeRobot v2.1** (parquet + mp4, `pick_trocar_teleop_success_train`)        |
| Env for eval        | Real LIBERO                                           | Real robot                                                                  |
| Task                | LIBERO Spatial/Object/Goal                            | 1 task: "pick trocar w/ left, handover, place on plate" (3 phases labelled) |

Final loss / advantage stays the same as Wan: **GRPO**, `loss_type: actor`,
`adv_type: grpo`, group-normalised relative advantages.

---

## 1. Locked-in facts (do not re-derive)

* **GR00T policy config** — `~/Isaac-GR00T` (branch `yunl/pick-trocar`),
  file `examples/g1-dex3/g1_dex3_head_config.py`.
  * video: 1 key `head_view` (maps to `observation.images.cam_head`, 480×640, 30 fps h264)
  * state: `left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7)` = **28-D flat**, `delta_indices=[0]`
  * action: same 4 keys, **`delta_indices=list(range(0, 16))` → chunk length 16 timesteps**
    * arms: `ActionRepresentation.RELATIVE` (delta joint targets)
    * hands: `ActionRepresentation.ABSOLUTE` (joint positions)
  * language: `annotation.human.task_description` (**NOT** `annotation.human.action.task_description` — LIBERO uses that variant, we do not)
* **Training script** — `~/Isaac-GR00T/run_head_10k.sh`
  * base: `nvidia/GR00T-N1.7-3B`
  * dataset: `/localhome/local-yunl/g1_pick_trocar_200_train0_190` (currently missing on this box, was on the source machine — need to relocate)
  * output: `/localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/head_10k`
* **Available LeRobot v2.1 datasets on the current machine** (`/localhome/local-yunl/data/`):
  * `pick_trocar_teleop_success_train`   — 225 eps, 34021 frames, 30 fps, **has manual phase annotations** in `meta/episodes.jsonl` under `source_annotation.cleaned_phase_frames` (`left_hand_pickup`, `handover_to_right_hand`, `placed_on_plate`) — free labels for the 4-class reward model
  * `pick_trocar_teleop_success_validation`
  * `pick_trocar_rollouts_10k_bs32_{train,val}`   — GR00T rollouts (contains failures)
  * `pick_trocar_rollouts_30k_bs256_{train,val}` — GR00T rollouts (contains failures)
* **Reward model** — 4-class progress classifier being trained on another machine by you. The `DreamDojoEnv` scaffold expects `reward_model.predict_rew(images) -> Tensor[B]` **already reduced to `[0, 1]`** (e.g. `(softmax(logits) * arange(4)).sum(-1) / 3.0` inside the model). Threshold for success = 0.9.
* **DreamDojo** — https://github.com/NVIDIA/DreamDojo. Not yet dropped into this repo. Cosmos-Predict2.5-based, autoregressive, 10 FPS × >60s stable generation, HF checkpoints `nvidia/DreamDojo` (2B + 14B post-trained).
* **Eval env** — real robot (out of scope for this repo; you have a separate bridge).

---

## 2. Files created / modified in the RLinf repo

All work happens on branch `main` in `/localhome/local-yunl/RLinf`. Uncommitted.

### 2.1 NEW  `rlinf/envs/world_model/world_model_dreamdojo_env.py`
Full `DreamDojoEnv(BaseWorldEnv)` scaffold, ~430 lines. Contract matches
`WanEnv` so `EnvWorker` and the GR00T actor can consume it unchanged.

**Ready:** reset / chunk_step / metric bookkeeping / GRPO group init sampling
/ image queue / condition_action buffer (28-D) / KIR wiring / offload/onload
stubs / obs wrapping (`states[:, :28]` zeros, `wrist_images=None`, matches
LeRobot obs shape) / auto-reset stub.

**TODO markers** (grep `TODO(dreamdojo)`):

1. **Imports** (top): replace placeholders with real DreamDojo pipeline & reward-model classes.
2. **`_build_pipeline`**: instantiate the DreamDojo generator on `self.device`.
3. **`_load_reward_model`**: return your 4-class progress classifier (must expose `predict_rew(imgs) -> [B] in [0,1]`).
4. **`_infer_next_chunk_frames`**: the AR generation call — build condition frames from `self.image_queue`, pass `action=actions_tensor` of shape `[B, T, 28]`, `prompt=self.task_descriptions`, unpack the returned video and update `self.image_queue` + `self.current_obs`.
5. **`_handle_auto_reset`**: currently resets all envs on any done; change to selective per-env reset if wanted.
6. **`offload` / `onload`**: move DreamDojo submodules CPU ↔ GPU.

### 2.2 NEW  `rlinf/data/datasets/lerobot_world_model.py`
`LeRobotV21InitDataset` — reads `pick_trocar_teleop_success_*` directly
(no offline `.npy` conversion needed). Emits the exact same dict schema as
the existing `NpyTrajectoryDatasetWrapper`:
`{start_items, target_items, task, episode_index, dataset_meta}`.

Key knobs (all YAML-overridable via `env.train.*`):
* `video_key="head_view"` — resolves the mp4 via `modality.json["video"][video_key]["original_key"]`
* `state_dim=28`, `action_dim=28` — asserts against the parquet shape
* `enable_kir=True`, `kir_context_len=4` — KIR context = 4 frames *after* the start frame (seeds WM with real dynamics before free-running)
* `image_size=(H, W)` — bilinear resize during load
* `episodes` — optional whitelist for train/val splits
* `random_start_frame=False` — set True to sample the start frame instead of always using frame 0
* `video_backend="decord"` (fallback `"pyav"`)

Also exposes `dataset_meta["phase_frames"]` and `dataset_meta["success"]` so
reward-model training and progress logging can consume them for free.

### 2.3 MODIFIED  `rlinf/models/embodiment/gr00t/simulation_io.py`
Added two functions (placed just above `convert_maniskill_obs_to_gr00t_format`):

* `convert_g1_dex3_wm_obs_to_gr00t_format(env_obs)` — WM obs → GR00T N1.7 g1_dex3 obs dict (`video.head_view`, `state.left_arm/right_arm/left_hand/right_hand`, `annotation.human.task_description`). Handles both torch tensors and numpy inputs. Hard-checks `states.shape[1] == 28`.
* `convert_to_g1_dex3_action_n1d7(action_chunk, chunk_size=16)` — GR00T action-dict → flat 28-D array in `left_arm, right_arm, left_hand, right_hand` order. Preserves the RELATIVE-arms / ABSOLUTE-hands split (caller must not blindly accumulate).

### 2.4 Reference: unchanged Wan pieces we're mirroring
* `rlinf/envs/world_model/base_world_env.py` — abstract base, defines the interface
* `rlinf/envs/world_model/world_model_wan_env.py` — canonical example (845 lines)
* `rlinf/data/datasets/world_model.py` — old NpyTrajectoryDatasetWrapper (kept, not deleted)
* `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml` — recipe to copy from
* `examples/embodiment/config/env/wan_libero_spatial.yaml` — env yaml to copy from
* `rlinf/workers/env/env_worker.py` — the worker that instantiates our env class via `get_env_cls`; no change needed
* `rlinf/algorithms/advantages.py` — GRPO advantage; no change needed

---

## 3. What is still TODO (in priority order)

### Phase A — get the env class runnable (before touching training)
1. **Register the new env** in `rlinf/envs/__init__.py`:
   * Add `DREAMDOJOWM = "dreamdojo_wm"` to `SupportedEnvType`.
   * In `get_env_cls()`, add `elif env_type == SupportedEnvType.DREAMDOJOWM: from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv; return DreamDojoEnv`.
2. **Add action-utils branch** in `rlinf/envs/action_utils.py::prepare_actions()`:
   * `elif env_type == SupportedEnvType.DREAMDOJOWM and wm_env_type == "trocar": return raw_chunk_actions` (pass-through — do NOT apply the LIBERO gripper `2x-1` hack; the RELATIVE-arms / ABSOLUTE-hands split is already correct at this point).
3. **Fill DreamDojo TODOs** in `world_model_dreamdojo_env.py` once you drop the `NVIDIA/DreamDojo` package into the venv (or vendor under `rlinf/models/embodiment/dreamdojo/`). Match the API shape from `WanEnv._infer_next_chunk_frames` — biggest thing is: build condition frames from `self.image_queue`, call `self.pipe(**kwargs)`, then update `image_queue` last 4 frames + `current_obs` sliding window.
4. **Drop-in reward model**: expose your 4-class classifier as a Python class with `predict_rew(images_bchw) -> tensor[B] in [0,1]` and wire it into `_load_reward_model`.

### Phase B — plumbing so `train_embodied_agent.py` accepts the new recipe
5. **Env config YAML** at `examples/embodiment/config/env/dreamdojo_trocar.yaml` — copy structure from `env/wan_libero_spatial.yaml`, set:
   * `env_type: dreamdojo_wm`
   * `wm_env_type: trocar`
   * `initial_image_path: /localhome/local-yunl/data/pick_trocar_teleop_success_train`  (LeRobot v2.1 dir, not `.npy`)
   * `video_key: head_view`
   * `chunk: 16`, `condition_frame_length: 5`, `num_frames: 21`
   * `image_size: [480, 640]` OR the resolution DreamDojo prefers (2B likely 256×256; 14B may go higher — pick and downscale on load)
   * `action_dim: 28`, `state_dim: 28`
   * `enable_kir: true`
   * `num_inference_steps: 3` (start low for wall-clock; increase later)
   * `reward_model.type: ProgressResnetRewModel`, `reward_model.from_pretrained: /path/to/your/rm.pth`
   * `reset_gripper_open: false`, `success_reward_threshold: 0.9`
6. **Recipe YAML** at `examples/embodiment/config/dreamdojo_trocar_grpo_gr00t.yaml` — copy from `wan_libero_spatial_grpo_openvlaoft.yaml`, change:
   * `defaults`: `env/dreamdojo_trocar@env.train`, `env/dreamdojo_trocar@env.eval` (or use a real-robot eval env once wired), `model/gr00t@actor.model`
   * `actor.model.model_type: gr00t_n1d7`, `model_path: /localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/head_10k`, `num_action_chunks: 16`, `action_dim: 28`, `obs_converter_type: g1_dex3_wm`
   * `algorithm.group_size: 8`, `reward_type: action_level`, `adv_type: grpo`, `loss_type: actor`
   * `env.train.total_num_envs: 8` initially (small — DreamDojo generation is expensive)
7. **Register the new obs_converter** — inside GR00T's actor worker, find where `obs_converter_type` dispatches (grep for `convert_libero_obs_to_gr00t_format`) and add a `g1_dex3_wm` case pointing at `convert_g1_dex3_wm_obs_to_gr00t_format`. Same for the action decoder (`convert_to_g1_dex3_action_n1d7`).

### Phase C — install / Docker (nice to have, blocks CI only)
8. `requirements/install.sh` — add `install_dreamdojo_world_model()` (clone `NVIDIA/DreamDojo`, `uv pip install -e .`, install deps in `requirements/embodied/models/dreamdojo.txt`) and an `--env dreamdojo` branch (and/or `--model dreamdojo_wm`). Follow the `install_wan_world_model()` pattern.
9. `docker/Dockerfile` — add a `embodied-dreamdojo` stage mirroring `embodied-wan`.

### Phase D — validation before real GRPO run
10. **Smoke test the init dataset**:
    ```bash
    cd /localhome/local-yunl/RLinf && python -c "
    from rlinf.data.datasets.lerobot_world_model import LeRobotV21InitDataset
    ds = LeRobotV21InitDataset(
        '/localhome/local-yunl/data/pick_trocar_teleop_success_train',
        video_key='head_view', image_size=(256, 256), enable_kir=True,
    )
    sample = ds[0]
    print('n_eps', len(ds), '| start frame shape', sample['start_items'][0]['image'].shape,
          '| KIR frames', len(sample['target_items']), '| task:', sample['task'][:60])
    print('state dim', sample['start_items'][0]['observation.state'].shape,
          '| action dim', sample['start_items'][0]['action'].shape)
    print('phase_frames', sample['dataset_meta'].get('phase_frames'))
    "
    ```
11. **Syntax check** (was interrupted before machine switch — repeat on the new host):
    ```bash
    python -c "
    import ast
    for p in ['rlinf/data/datasets/lerobot_world_model.py',
              'rlinf/envs/world_model/world_model_dreamdojo_env.py',
              'rlinf/models/embodiment/gr00t/simulation_io.py']:
        ast.parse(open(p).read()); print('OK', p)
    "
    ```
12. **Single-env dry run** once DreamDojo TODOs are filled — instantiate `DreamDojoEnv(cfg, num_envs=1, ...)`, call `env.reset()`, feed a random `[1, 16, 28]` action into `env.chunk_step(...)`, confirm shapes.

---

## 4. Quick resume on the new machine

Assuming the new machine has SSH access to your `origin` and can see or replicate the datasets:

```bash
# 1. RLinf side (this repo, uncommitted work)
cd ~/RLinf   # or wherever you clone
git status   # should show:
#   modified:  rlinf/models/embodiment/gr00t/simulation_io.py
#   untracked: rlinf/data/datasets/lerobot_world_model.py
#   untracked: rlinf/envs/world_model/world_model_dreamdojo_env.py
#   untracked: DREAMDOJO_HANDOFF.md   (this file)
# If uncommitted work isn't on the new host, copy those 4 files over via scp / rsync.

# 2. GR00T side
cd ~/Isaac-GR00T
git remote -v         # should be KumoLiu/Isaac-GR00T
git checkout yunl/pick-trocar
# Then re-run  bash run_head_10k.sh  after fixing the dataset path.

# 3. DreamDojo (still to install)
git clone https://github.com/NVIDIA/DreamDojo ~/DreamDojo
# (checkpoints from HF: nvidia/DreamDojo)

# 4. Datasets — the LeRobot trees under /localhome/local-yunl/data/ need to be
#    available at the same path (or symlinked, or the YAML path adjusted).
#    Same for the reward-model checkpoint from the other training machine.
```

Then resume at **Phase A step 1** in §3.

---

## 5. Assets referenced (for provenance)

* Wan example doc:  `docs/source-en/rst_source/examples/embodied/wan.rst`
* Wan env impl:     `rlinf/envs/world_model/world_model_wan_env.py`
* Wan recipe YAML:  `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`
* GR00T fork:       https://github.com/KumoLiu/Isaac-GR00T (branch `yunl/pick-trocar`)
* Modality cfg:     `~/Isaac-GR00T/examples/g1-dex3/g1_dex3_head_config.py`
* Train script:     `~/Isaac-GR00T/run_head_10k.sh`
* DreamDojo:        https://github.com/NVIDIA/DreamDojo
* Datasets (this host):  `/localhome/local-yunl/data/pick_trocar_{teleop_success,rollouts_*}_*`
* Reward-model training: happening on a separate machine (you)

---

## 6. Decisions still open

* **Selective vs full auto-reset** on chunk termination (currently full).
* **Image resolution for training** — 256² (fast, Wan-style) vs 480×640 native (matches teleop data, more VRAM).
* **KIR direction** — this scaffold puts KIR context AFTER the start frame (frames `start+1..start+4`, seeds WM with real dynamics before free-running). Wan's docstring said "last 4 frames" which is ambiguous; if you prefer BEFORE the start frame, change the two lines in `LeRobotV21InitDataset.__getitem__`:
  ```python
  wanted_indices.extend(start_frame + 1 + i for i in range(self.kir_context_len))
  # -> replace with:
  wanted_indices.extend(start_frame - self.kir_context_len + i for i in range(self.kir_context_len))
  ```
  and adjust the `max_start` / `random_start_frame` bounds accordingly.
* **Whether to also feed the last-known state** into `_wrap_obs()` (right now it's zeros because DreamDojo doesn't produce proprio). If GR00T's g1_dex3 config was trained *with* proprio, you may want to at least seed the first frame's state from the init dataset and freeze it thereafter, or run a lightweight state estimator alongside the WM.

---

_Last touched: 2026-09-09._
