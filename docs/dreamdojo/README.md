# DreamDojo × GR00T N1.7 原生 GRPO

只保留一个训练配方：
`examples/embodiment/config/dreamdojo_trocar_grpo_gr00t_n1d7.yaml`。
train、定期 eval、checkpoint 均由 RLinf 原生 runner 执行；没有额外的评测进程池。

## 当前默认设置

- 从 GR00T SFT base 初始化，不 resume；Flow-SDE `noise_level=0.1`，physical action noise=0。
- WM train/eval 均为 35 步降噪，GR00T 为 4 步；此处没有改实验算法或奖励定义。
- 7 个 GPU worker，各 8 个 train env；group=8、global batch=56、micro batch=2。
- 每轮执行 240 个 30 Hz 动作，输出 120 个 WM 帧，加初始帧的视频为 15 fps。
- 55 个固定验证案例，56 个计算槽位，padding 不计入 eval；每 5 轮 eval 和保存。
- 默认上限 1000 轮。最近完成的 noise=0.1 trial 是启动时 CLI 指定的 5 轮，并非 YAML 上限。
- train/eval 视频均开启。WM 中的成功率是 reward classifier 的判定，不等同于真实机器人成功率。

本机默认排除**物理 GPU 4**；换机器时需明确配置设备编号，不能沿用这个故障判断。

actor micro batch 的独立性能测试与边界见 [ACTOR_MICROBATCH_BENCHMARK.md](ACTOR_MICROBATCH_BENCHMARK.md)。

## 路径和环境

`run_dreamdojo_trocar.sh` 只是原生 Python 入口的环境包装器，不负责循环、评测调度或早停。
本机沿用 sibling `DreamDojo/`、`data/`、`models/`；cluster 的独立 Docker/sqsh 环境见
[CLUSTER.md](CLUSTER.md)，容器内路径见 [cluster.env.example](cluster.env.example)。
直接运行包装器时可显式选择 site 文件：

```bash
DREAMDOJO_SITE_ENV=/absolute/path/site.env \
  bash examples/embodiment/run_dreamdojo_trocar.sh --cfg job --resolve
```

这条命令只解析配置，不分配 GPU 任务。解释器选择顺序：`RLINF_PYTHON` → 已激活环境 →
RLinf `.venv` → PATH 中的 python。`DREAMDOJO_SITE_ENV` 仅在显式指定时 source；
诊断 Python 工具需先自行 source 同一个文件。

| 环境变量 | 内容 |
| --- | --- |
| `DREAMDOJO_ROOT` | 已打补丁的 DreamDojo 仓库 |
| `DREAMDOJO_DATA_ROOT` | 六个原始 LeRobot v2.1 数据集的共同父目录 |
| `GR00T_MODEL_PATH` | 完整 SFT checkpoint 目录，包含 processor/config/statistics |
| `DREAMDOJO_WM_CHECKPOINT` | 当前 LoRA r64、iter18000 的完整 EMA `.pt` |
| `DREAMDOJO_REWARD_CHECKPOINT` | milestone v2 `best.pt` |
| `DREAMDOJO_LAM_CHECKPOINT` | `LAM_400k.ckpt` 的实际路径；严格加载，缺失不再静默继续 |
| `DREAMDOJO_ACTION_STATISTICS` | 与 WM 训练一致的 `G1_stats.json` |
| `DREAMDOJO_GPUS` | 原生 actor/env/rollout 共享的设备 placement |
| `HF_HOME` | DreamDojo/GR00T 所需的预下载 HF 资产缓存 |

数据集目录名：

```text
pick_trocar_teleop_success_train
pick_trocar_rollouts_30k_bs256_train
pick_trocar_rollouts_10k_bs32_train
pick_trocar_teleop_success_validation
pick_trocar_rollouts_30k_bs256_val
pick_trocar_rollouts_10k_bs32_val
```

RL reset 读取原始 28D 数据；不要误用旧 DreamDojo Slurm 脚本同步的 `g1_hf_*` 适配后数据。
数据的 parquet、meta 和视频必须一起迁移，确认视频软链接在目标机可解析。
权重、数据、HF 缓存与源码分开同步，不上传日志、旧实验输出、`.venv` 或仓库外归档。

DreamDojo 使用其自身的视频 tokenizer/文本编码器；仅拷贝 WM EMA 文件未必足够。
默认 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，目标机须事先准备缓存。
其实验注册还依赖 `configs/2b_480_640_g1_hf_teleop_rollout_posttrain_lora_lr3e-4_r64.yaml`；
该文件目前是未跟踪文件，迁移/提交时不能漏掉。WM 训练 YAML 中的训练数据路径不被 RL rollout 迭代；
若要在 cluster 重训 WM，需要另外调整那些路径。

不要直接把 DreamDojo 的 `dreamdojo_train.slurm` 当作 RL 启动脚本：它是 WM 训练容器/挂载方案，
不保证包含 RLinf、GR00T 和所需版本。独立容器及单节点原生短训练已验证，
当前镜像补充方式和完整 eval 进度见 [CLUSTER.md](CLUSTER.md)；多节点尚未验证。
不得清空 Slurm 的 GPU 限制来访问未获分配的设备；placement 编号须与实际 RLinf 节点发现一致。

## 多组对比实验

新旧 WM（rank 64 与 `scratch_lr3e-4` rank 32）的路径、配套 experiment 和对比方法见
[WM_COMPARISON.md](WM_COMPARISON.md)；切换时必须同时选择权重和匹配的 WM 结构。
九组曲线、停止/保留决策与下一步建议见 [SWEEP_REVIEW_20260912.md](SWEEP_REVIEW_20260912.md)。
补 seeds、noise/LR 邻近参数与 15 步 WM 对照见 [FOLLOWUP_SWEEP_20260912.md](FOLLOWUP_SWEEP_20260912.md)。

cluster 的可选自动续跑入口是 `docker/dreamdojo/train_chain.slurm`，默认单份作业 4 小时、
`CHAIN_TIMEOUT=3.9h`；预算不足时按代保存、自动提交并恢复原生训练状态。
提交方法、最大链长、停止方法及真实验证记录见 [SLURM_CHAIN.md](SLURM_CHAIN.md)。

每个实验使用独立 log path，通过 CLI 覆盖参数，不复制 YAML，也不加载修复前的 step20。
例如已分配的单节点上，比较 noise 时：

```bash
DREAMDOJO_SITE_ENV=/absolute/path/site.env \
  bash examples/embodiment/run_dreamdojo_trocar.sh \
  runner.logger.log_path=/absolute/path/runs/noise01_seed0 \
  runner.logger.experiment_name=noise01_seed0 \
  actor.model.rl_head_config.noise_level=0.1

DREAMDOJO_SITE_ENV=/absolute/path/site.env \
  bash examples/embodiment/run_dreamdojo_trocar.sh \
  runner.logger.log_path=/absolute/path/runs/noise05_seed0 \
  runner.logger.experiment_name=noise05_seed0 \
  actor.model.rl_head_config.noise_level=0.5
```

上面是两个独立任务示例，不要在同一组已占用 GPU 上同时启动。可对独立训练重复覆盖
`actor.seed`、`env.train.seed`、`env.eval.seed`，并记录这三个种子；噪声对比时保持种子、
batch、初始数据混合、WM 降噪步数和 eval 协议相同。不要把不同 batch/并行度的结果称为严格配对。

改变训练 WM 精度只覆盖 `env.train.num_inference_steps=5`，统一保留
`env.eval.num_inference_steps=35` 作为比较协议。关闭训练视频用
`env.train.video_cfg.save_video=false`。短 smoke 用 `runner.max_epochs=5`，默认每 5 轮 eval。

如果 cluster 要用整节点 8 卡，可以设置 `DREAMDOJO_GPUS=0-7`，并覆盖：

```text
env.train.total_num_envs=64 actor.global_batch_size=64 env.eval.total_num_envs=56
```

此时 train 为 8×8，eval 为 8×7，仍覆盖 55 个唯一案例。eval 的分片计数已有 CPU 测试；
8 卡完整 GPU eval 已在 job 1104331 验证，见 [CLUSTER.md](CLUSTER.md)。
若要直接与本机历史 56-env 结果比较，先保持 7 个 worker。

原生仅评测 SFT base：使用仓库已有 `evaluations/eval_embodied_agent.py`，通过
`--config-path` / `--config-name` 选择上面的主 YAML，覆盖 `runner.only_eval=true`、
`'rollout.model=${actor.model}'` 并使用新的 log path。不要直接给训练入口加 only_eval
就认为它会调用独立评测 runner；当前训练包装器只启动训练。具体运行记录见
[GR00T_UPGRADE.md](GR00T_UPGRADE.md)。
当前配置不加载 RL checkpoint；评测某个 RL checkpoint 时需另行明确 checkpoint 加载参数。

## 回归与离线诊断

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit_tests/test_dreamdojo_adapters.py \
  tests/unit_tests/test_dreamdojo_native_eval.py \
  tests/unit_tests/test_dreamdojo_inference_perf.py \
  tests/unit_tests/test_dreamdojo_diagnostics.py
```

cluster 改用自己的 Python。`DREAMDOJO_ROOT` 缺失时只跳过需要外部源码的测试，不算完成跨仓库验证。

- `toolkits/world_model/dreamdojo_validation.py`：parity、真实动作 WM 的 teacher-forced/closed-loop、
  policy、真实视频 reward 以及结果报告。`--help` 列出当前命令；不再支持 snapshot/grpo。
- `toolkits/world_model/dreamdojo_perf.py`：仅保留同输入 WM 和 policy 推理计时/一致性检查，
  不再启动另一套训练。仅在空闲且获分配的单 GPU 上使用。
- 诊断只保存参数、结果、必要视频/数组和计时，不再自动复制源码或 `.diff`。
  版本追踪使用 Git 和原生保存的 resolved config；历史日志未改动。

整理决策及本次验证见 [CLEANUP.md](CLEANUP.md)。历史实验证据见 [history/](history/)。
