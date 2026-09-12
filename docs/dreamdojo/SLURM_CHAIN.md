# Slurm 自动续跑（2026-09-11）

已实现、同步 cluster，并通过真实两段自动接续验证：`1105946 → 1106030`。
两段均 COMPLETED 0:0，最终 step 3 保存后正常结束，未提交第三份作业。

参考 `DreamDojo/dreamdojo_sweep_train.slurm` 及 `scripts/lib/chain.sh`。
新增 `docker/dreamdojo/train_chain.slurm`，默认每份作业 4 小时、单节点 8 GPU，
`CHAIN_TIMEOUT="${CHAIN_TIMEOUT:-3.9h}"`。仍调用 `run_cluster.slurm train` →
`run_dreamdojo_trocar.sh` → 原生 `train_embodied_agent.py`；没有新增训练器或外部 eval。
原来的 smoke、verify、eval、单次 train 入口不会自动续提交。

## 为什么不能只在 3.9 小时 kill

当前 64 env × 2 rollout 的一代约 18 分钟，4 小时减 3.9 小时只剩 6 分钟。
直接 kill 可能丢掉从上次 `save_interval=5` 到现在的多代更新。
本实现把 3.9 小时作为本份作业的**运行预算上限**：每代开始前预估下一代、到期的原生 eval、
保存所需时间；预计来不及就停在已经完成的代末，强制保存再退出。
因此正常情况下可能早于 3.9 小时交接，不会等到最后 6 分钟才开始做 18 分钟的工作。

初始保守预算：训练一代 1200 秒、一次 eval 600 秒、保存 120 秒；
训练/eval 额外留 20% 余量，实测更慢时上调，并随 checkpoint 传给下一份作业。
可用 `CHAIN_STEP_SECONDS`、`CHAIN_EVAL_SECONDS`、`CHAIN_SAVE_SECONDS` 调整初始估计。
Slurm 已用时间（含容器/模型启动）计入预算；`CHAIN_TIMEOUT` 至少比 Slurm `--time` 少 90 秒。
这些是时间估计，不是硬实时保证。外层仍有 timeout 兜底，异常变慢时只从最近一次已完成保存恢复，
可能丢失该 checkpoint 之后的工作。若初始化后连一代都放不下，报错停止，不循环占用 GPU。

## 保存、恢复和退出规则

- 原生 actor 的 FSDP DCP 保存完、所有 rank 的保存调用返回之后，才原子发布
  `global_step_N/training_state.json` 和 chain 的 `latest.json`。
  不扫描“名字最大”的目录，不加载保存中断留下的半成品。续跑入口校验标记、元数据及 8 个分片。
- 下一份作业追加 `runner.resume_dir=/outputs/.../global_step_N`。
  原生恢复模型、optimizer、LR scheduler、actor RNG，并从 global step N 继续；不是只加载权重。
  checkpoint 仍采用原生布局，并保留 `full_weights.pt`，可用于独立 eval。
- `val_check_interval=5`、`save_interval=5` 按全局代数继续，不因重启重新计数；到期 eval 不被跳过。
  预算交接额外保存一次；最终完成也确保保存。正常完成后不续提交。
- 原生 checkpoint **没有保存 env/rollout 采样器的随机状态和环境内部状态**，重启会重新初始化这些进程。
  不保证与不中断训练逐轨迹/逐 bit 一致，也不能把链式运行直接当成严格确定性的 A/B。
  本实现不更改采样分布、GR00T 去噪或 WM conditioning。
  另外，原生 manager 的 Python `optimizer_steps`（不是 Adam state 中的 step）未保存；当前无 critic warmup，
  不影响这份 GRPO 配方。若改用 critic warmup / AMP scaler 等其他训练配方，需要单独核验其恢复状态。
- 普通 Python/CUDA/NCCL 错误、手动 `scancel` 不自动重试。只有正常预算交接，或 timeout 且有新完成的
  checkpoint，才会续提交。无训练进展不重试。新作业依赖 `afterok:父作业`，父作业取消不会启动子作业。
- 默认 `MAX_RUNS=6`（包含第一份作业），作为误操作保护；不足以跑完 1000 代时须显式提高。
  达到上限但尚未训练完会报错并保留 checkpoint，不伪装成训练完成。
  checkpoint 不自动删除；当前每次原生保存约 20.34 GB，长链/多实验需考虑存储空间。
- 在 chain 目录创建 `STOP`：在下一个代边界保存并停止，不再提交。
  紧急停止用 `scancel`；若子作业已排队，也取消它。STOP 可以让已排队的子作业启动后直接退出。

支持当前同步 FSDP 配方、8 GPU、`weight_sync_interval=1`、无 bootstrap prefetch；
不支持训练 pipeline 或任意变更 world size 的续跑。保留 partition/account/qos/time/job-name 和显式节点约束；
不是任意 Slurm 资源规格的通用重建器。同一 chain 的 Hydra 参数、主要资产环境变量、续跑源码 hash 必须相同，
更改实验配置请新建 chain。参数以 argv 数组传递，列表里的逗号及空格不会被 `--export` 拆坏。

## 提交示例

当前 v1 sqsh 内是旧 runner；必须显式只读挂载版本化的两个新源码文件。
这与 LAM 的独立源码补充分开管理，不覆盖镜像，不把 source/.diff 放进每个实验 log。

```bash
cd /lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf
source docs/dreamdojo/cluster.env.example
export RLINF_CHAIN_SOURCE_ROOT=/lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf-runtime/20260911-chain-v1
export CHAIN_TIMEOUT=3.9h
export MAX_RUNS=6   # 明确需要跨越更多 4h allocation 时再提高
sbatch docker/dreamdojo/train_chain.slurm \
  runner.logger.experiment_name=trocar_env64x2_mb8_noise01 \
  env.train.total_num_envs=64 env.train.rollout_epoch=2 \
  actor.global_batch_size=128 actor.micro_batch_size=8
```

这只是提交示例，**没有执行这份长训练或大规模 sweep**。正式 YAML 的 1000 代、val/save=5 不变。
第一份作业 ID 自动作为 `CHAIN_ID`；不同实验分别 sbatch，即各自独立续跑。
首次也可显式传入已有原生 `runner.resume_dir`，它必须指向容器可见的 `global_step_N` 目录；
后续自动续跑只选该 chain 自己发布完成标记的 checkpoint，不覆盖旧实验。

结果布局（宿主机 `outputs/rlinf` 映射到容器 `/outputs`）：

```text
outputs/rlinf/
  chains/<CHAIN_ID>/          # 少量交接元数据、后续作业的 Slurm stdout/stderr
    launch.json              # 参数 + 源码哈希，不复制源码；不是训练 checkpoint
    latest.json
    result-<JOB_ID>.json
    submitted-<JOB_ID>.json
  <JOB_ID>-train/             # 每个 allocation 独立日志/视频/TensorBoard，不互相覆盖
    <experiment>/checkpoints/global_step_N/actor/...
```

## 修改与验证记录

- `rlinf/utils/train_budget.py`：可选时间预算、完成标记及交接结果。
- `rlinf/runners/embodied_runner.py`：仅开启预算时添加整代保存退出；保留原生更新、eval、DCP。
  同时修复退出时先停止日志线程再 `queue.join()` 可能卡住的问题，并确保日志处理异常也会 `task_done()`。
- `docker/dreamdojo/{train_chain.slurm,chain.py,run_cluster.slurm}`：Slurm 提交、原子元数据、显式版本化源码挂载。
- `tests/unit_tests/test_dreamdojo_chain.py`：CPU 预算/保存失败/两次接续/参数保留/错误不续跑等测试。
- 本地全部 DreamDojo CPU 测试：97 passed（其中续跑测试 32 项）；Ruff、`bash -n`、`git diff --check` 通过。

### 真实 cluster 短测

测试只验证续跑，不用于比较 success/loss；轨迹缩短为 12 actions，关闭 eval 与周期保存，
从已完成的有效实验 `1105305` 的 step 1 恢复。这样可以验证交接时的**强制保存**，
同时原始 optimizer 含已有 20 次更新，不是从零状态开始的空恢复。

```bash
export CHAIN_TIMEOUT=8m
export CHAIN_STEP_SECONDS=120 CHAIN_EVAL_SECONDS=60 CHAIN_SAVE_SECONDS=60
export MAX_RUNS=2
sbatch --time=00:10:00 --job-name=rlinf-dd-chain-check \
  docker/dreamdojo/train_chain.slurm \
  runner.max_epochs=3 runner.val_check_interval=-1 runner.save_interval=-1 \
  runner.resume_dir=/outputs/1105305-train/cluster_env64x2_mb8_lam_p2p/checkpoints/global_step_1 \
  runner.ckpt_path=null runner.logger.experiment_name=cluster_chain_resume_check \
  env.train.total_num_envs=64 env.train.rollout_epoch=2 \
  env.train.max_episode_steps=12 env.train.max_steps_per_rollout_epoch=12 \
  actor.global_batch_size=128 actor.micro_batch_size=8
```

| 环节 | Job / 节点 | 状态 |
| --- | --- | --- |
| 原始有效 checkpoint | `1105305` / `pool0-00318` | step 1；抽查 Adam step=20、LR last_epoch=1 |
| 第一段 | `1105807` / `pool0-00208` | COMPLETED 0:0，7m09s；从 step 1 更新到 2，Adam step=21、LR last_epoch=2；自动提交 `1105903` |
| 自动接续第二段 | `1105903` / `pool0-00063` | FAILED 1:0，4m40s；已加载 step 2，但初始化后剩余预算不足以再开始一代，主动报错；未自动重试 |
| 9 分钟重测第一段 | `1105946` / `pool0-00020` | COMPLETED 0:0，7m04s；step 1 → 2；正常保存后自动提交 `1106030` |
| 9 分钟重测自动第二段 | `1106030` / `pool0-00308` | COMPLETED 0:0，7m10s；读取前一段 step 2，更新并保存 step 3；无第三份作业 |

第一次日志明确记录 `Chain budget: saving and handing off at step 2`；
交接采用正常退出（exit_code=0），不是依赖 kill 丢掉当前代。
`latest.json` / `result-1105807.json` / `submitted-1105807.json` 位于 cluster
`outputs/rlinf/chains/1105807/`。每段输出仍在各自的 `<JOB_ID>-train/`。
本地小文件归档：`logs/cluster_chain_20260911/`；没有下载约 20 GB 的完整权重，
只下载 DCP 元数据并按 offset 读取少量状态字段以核验 optimizer/LR 的接续。

8 分钟试验中，观测训练一代为 123.09 秒，因此下一代加保存的预估为
`1.2 × 123.09 + 60 = 207.71` 秒；第二份作业初始化后剩余时间略低于这个门槛。
这是刻意短测的预算问题，不是 DCP 加载失败，也没有修改生产的 3.9 小时参数或安全检查来绕过它。
后续 `1105946` 仅把 `CHAIN_TIMEOUT` 改为 `9m`、Slurm 时间改为 `00:11:00`、
实验名改为 `cluster_chain_resume_check9m`，其他训练参数与上述命令一致；完整两段流程通过。

成功链的元数据：`outputs/rlinf/chains/1105946/`；最终
`result-1106030.json` 为 `status=complete, global_step=3, max_steps=3`，
日志明确记录 `Native training completed; no continuation`，队列已空。
最终 checkpoint：
`outputs/rlinf/1106030-train/cluster_chain_resume_check9m/checkpoints/global_step_3/`。

对原始 step 1、成功链 step 2、step 3 的 DCP 做实际字段抽查（只额外读取 37,308 字节）：

| 状态 | 原始 `1105305` | 第一段 `1105946` | 第二段 `1106030` |
| --- | --- | --- | --- |
| Adam step（三个不同参数） | 20 | 21 | 22 |
| LR scheduler `last_epoch` | 1 | 2 | 3 |
| LR scheduler `_step_count` | 2 | 3 | 4 |
| LR | 5e-6 | 5e-6 | 5e-6 |
| 抽查 bias 分片的一阶动量 norm | 1.54342415e-4 | 1.38937990e-4 | 1.25079387e-4 |

抽查参数是 `action_head.model.timestep_encoder.timestep_embedder.linear_1.bias` 的首个 192 元素分片；
短轨迹没有任务奖励，Adam 的已有一阶动量仍按 beta1=0.9 连续衰减，两次逐元素比较最大误差均为 0。
二阶动量也保持非零。结合 DCP load/save 实际成功，确认不是只恢复模型权重后重新建零状态 optimizer。
这不是对所有参数逐字节重演或确定性轨迹的证明。

测试过程没有更改 runtime 补丁；成功链的 `launch.json` 保留三个 Slurm 入口文件及两个只读 runtime 文件的 SHA256。
本机与 cluster 使用同一组源码。9 分钟仅为验收加速参数，默认入口依然是 4 小时 allocation / 3.9 小时预算；
正式 YAML 仍为 1000 代、val/save=5。尚未提交正式长训练或大规模对比实验。
