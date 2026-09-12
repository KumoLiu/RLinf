# Cluster 总采样量扩展检查（2026-09-11）

用户批准测试 128 env；如果当前 8×H100 放不下，则保留 64 个并行 env、分两轮采样。
本次只做原生单代短测，不启动长训、不改正式 YAML，不引入另一套 train/eval。
沿用现有 v1 sqsh + 只读 LAM 源码补充、SFT base、35 步 WM、4 步 policy、noise=0.1、
原有动作桥/reward、240 条命令/轨迹、保存 train 视频及单代 checkpoint；本次禁用完整 eval。
本次 128-env 首轮没有改变 LAM 加载基线。后续用户优先要求修复 LAM 并恢复 cluster P2P，
当时暂停了 64×2 备选；修复与验证另见 [LAM_CHECKPOINT_AUDIT.md](LAM_CHECKPOINT_AUDIT.md)。
LAM/P2P 验证完成后，用户再次授权继续，文末 job `1105305` 已完整通过。

## 三种配置的含义

| 配置 | 已通过的基线 | 本次先测 | 显存不足时的备选 |
| --- | --- | --- | --- |
| `env.train.total_num_envs` | 64 | 128 | 64 |
| `env.train.rollout_epoch` | 1 | 1 | 2 |
| 每代轨迹总量 | 64 | 128 | 128 |
| 每卡同时运行的 env | 8 | 16 | 8 |
| 每代 GRPO 组数（每组 8 条） | 8 | 16 | 16 |
| `actor.global_batch_size` | 64 | 128 | 128 |
| `actor.micro_batch_size` | 8 | 8 | 8 |
| 每次 optimizer step 的梯度累积次数 | 1 | 2 | 2 |

每条轨迹为 20 个 action chunk；原生 actor 将时间×轨迹展平为训练样本。
基线每代 1,280 个片段，后两者为 2,560 个；默认 update_epoch=1 时三者均为 20 次全局
optimizer step。global batch 128 不是每次更新 128 个完整视频；micro 8 也不限制 WM batch。
初始 case 按训练分布抽样，16 组不保证是 16 个互不重复的 case。

备选的优势只是 WM 同时运行数量不变；总采样耗时预计接近两倍，缓存轨迹/forward inputs
更多，actor 梯度累积也可能改变峰值，因此同样需要完整短测，不能预先声称绝无 OOM。
比较成功率时还需考虑抽样、随机数、每代数据量不同，不能把本次差异直接归为大 batch 的收益。

## 128 env 首轮：1104860

节点 `pool0-00169`，1 节点 / 8 GPU，`batch` 分区、最长 1 小时。
此前中断的用户 turn 未留下重复作业；提交前 `squeue -u yunl` 为空。

在 cluster `code/RLinf` 下，设置与 [CLUSTER.md](CLUSTER.md) 相同的只读 LAM supplement、
`DREAMDOJO_GPUS=0-7` 和现有 SFT 路径，实际提交：

```bash
sbatch --partition=batch --job-name=rlinf-dd-env128-mb8 --time=01:00:00 \
  --output=/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/cluster_batch128_20260911/%x-%j.out \
  --error=/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/cluster_batch128_20260911/%x-%j.err \
  docker/dreamdojo/run_cluster.slurm train \
  runner.max_epochs=1 runner.val_check_interval=-1 runner.save_interval=1 \
  runner.resume_dir=null runner.ckpt_path=null \
  runner.logger.experiment_name=cluster_env128_mb8 \
  env.train.total_num_envs=128 env.train.rollout_epoch=1 \
  actor.global_batch_size=128 actor.micro_batch_size=8
```

注意入口内部的 64-env 默认值被末尾 CLI 覆盖，最终仍需核对保存的解析后 config。
原生输出：`outputs/rlinf/1104860-train`；Slurm 日志在 `outputs/rlinf/cluster_batch128_20260911/`。
本地记录：`logs/cluster_batch128_20260911/1104860/`。
同一已分配节点内每 5 秒采样一次 `nvidia-smi`；该只读 monitor 不申请额外 GPU。

实际结果：`FAILED / 255:0`，5 分 19 秒；首个 WM chunk 的 tokenizer 内发生 CUDA OOM，
尚未完成 rollout、actor 更新或 checkpoint 保存。解析后的 config 已确认 128 env、
global batch 128、micro batch 8、rollout_epoch=1 均生效。

根因位于主 stdout 第 2622 行：`torch.OutOfMemoryError`，申请 12.40 GiB 时只剩 11.36 GiB；
堆栈进入 `cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py` 的 `F.pad`。
这是 16 env/卡下的视频 tokenizer 内存不足，不是 LAM 加载或 actor micro batch 引起的报错。
随后的 Ray actor kill / NCCL 断连为退出清理阶段的连带日志，不能单独当作网络故障。
每 5 秒整卡采样的最大值为约 75.32 GiB；采样值并非精确 CUDA allocator 峰值。
失败日志和 config 已归档本地，原始失败记录保留，没有删除。

### 备选提交状态与 NCCL 注意事项

准备提交 `64 env / rollout_epoch=2 / global batch=128 / micro batch=8` 时用户中断了工具调用。
重新核对 `squeue` 和 `sacct` 后确认备选**尚未实际提交**；此前“已提交”的 commentary
未获作业号确认，应以此处核验结果为准。LAM/P2P 独立验证作业不属于这个备选实验。

用户随后指出本机 NCCL 故障规避不应照搬 cluster。核查发现 cluster 的
`run_cluster.slurm:59`、`cluster.env.example:23` 仍默认 `NCCL_P2P_DISABLE=1`，
容器内训练包装器也有相同回退值。后续已将 cluster 默认改为显式 0，并保留本机默认 1；
所检查的部署文件没有显式设置 `NCCL_P2P_LEVEL`。独立 8 卡通信验证记录见上述 LAM 审计文档；
这项改变不能直接解决本次 tokenizer OOM，也不能把旧耗时视为默认 P2P 下的性能。

## 64 env × 2 轮采样：1105305

用户在 LAM/P2P 验证通过后授权继续。提交前 `squeue -u yunl` 为空，
LAM 源码和 Slurm 启动脚本 SHA256 与已验证版本一致；共享权重仍为原文件。
这次是完整单代训练容量测试，不是 job 1105150 的单 chunk 最小测试，也不启动长训。

在 cluster `code/RLinf` 下 source `docs/dreamdojo/cluster.env.example`，
`unset NCCL_P2P_LEVEL`，使用 `NCCL_P2P_DISABLE=0`，提交：

```bash
sbatch --partition=batch --time=01:00:00 --job-name=rlinf-dd-env64x2-mb8 \
  --output=/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/cluster_batch64x2_20260911/%x-%j.out \
  --error=/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/cluster_batch64x2_20260911/%x-%j.err \
  docker/dreamdojo/run_cluster.slurm train \
  runner.max_epochs=1 runner.val_check_interval=-1 runner.save_interval=1 \
  runner.resume_dir=null runner.ckpt_path=null \
  runner.logger.experiment_name=cluster_env64x2_mb8_lam_p2p \
  env.train.total_num_envs=64 env.train.rollout_epoch=2 \
  env.train.max_episode_steps=240 env.train.max_steps_per_rollout_epoch=240 \
  actor.global_batch_size=128 actor.micro_batch_size=8
```

使用 SFT base，不加载旧 step20 或上次最小短测的结果；35 步 WM、4 步 policy、noise=0.1、
group=8 不变。每卡同时 8 env，顺序采样两轮；每代 128 条完整轨迹、16 个 GRPO 组，
每条 20 chunk（240 条 30-Hz 动作），共 2,560 个 actor 训练片段。
每 rank 的 global batch 份额为 16，分两次 micro batch 8 累积，再 optimizer step；
默认 update_epoch=1 下共 20 次 optimizer step。第二轮采样不提前更新 policy。
这些是 CLI 覆盖，没有修改正式 YAML 的长期设置；保存 train 视频及单代 checkpoint，不跑完整 eval。

原生 runner 输出：`outputs/rlinf/1105305-train`。
Slurm 日志和 GPU 采样：`outputs/rlinf/cluster_batch64x2_20260911/`。
GPU monitor 是同一已分配作业内的只读 overlap step，每 5 秒采样一次，不额外申请 GPU；
采样最大值不等于 CUDA allocator 的精确峰值。作业运行于 `pool0-00318`，8×H100。

验收项：实际 config 为 64/2/128/8；8 个 EnvGroup 严格恢复 LAM；两轮完整 rollout
和 actor 更新均完成；有 128 条轨迹的汇总指标、有非零/有限的训练信号、有完整 checkpoint；
检查第一轮与第二轮之间的显存变化及最终采样峰值。单代成功率不是训练后 eval，
也不能据此宣称更大 batch 改善效果；旧 64×1 的 P2P/LAM 加载设置不同，不是严格性能 A/B。

### 实际结果：通过

Slurm 主作业、batch、extern、容器训练 step 和 GPU 只读 monitor step 均为
`COMPLETED / 0:0`。主作业 22 分 20 秒，容器训练 step 22 分 14 秒；结束后队列为空。
保存的原生 config 已确认 64 env、rollout_epoch=2、global batch=128、micro batch=8，
每条轨迹 240 动作、WM 35 步、noise=0.1，完整执行到 `Global Step=1/1`。
8 个 EnvGroup 的 `rank_*.log:137` 均严格恢复 LAM，0 missing / 0 unexpected keys；
容器日志为 `NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=unset`。

| 指标（原生 TensorBoard 精确值） | 结果 |
| --- | --- |
| `env/num_trajectories` | 128 |
| `env/success_once` | 0.390625，50/128（39.0625%） |
| `env/picked` / `env/handed` | 0.75 / 0.4609375（96/128、59/128） |
| `env/return` | 1.6015625 |
| `rollout/group_return_std_mean` | 0.7244581580 |
| `rollout/group_flat_fraction` / `group_zero_fraction` | 0.125 / 0.0 |
| `train/actor/grad_norm` | 1.1621516943 |
| `train/actor/approx_kl` | 0.0014689212 |
| `train/actor/clip_fraction` | 0.0053647049 |
| `train/actor/policy_loss` | 0.0003099381 |
| `train/actor/total_loss` | 0.0001549691 |
| `time/generate_rollouts` | 967.2463 秒，约 16 分 7 秒 |
| `time/actor_training` | 64.6674 秒 |
| `time/sync_weights` | 17.5439 秒 |
| `time/step` | 1080.8301 秒，约 18 分 1 秒 |

第一轮 rollout 的进度记录为 495.11 秒（约 8 分 15 秒）；两轮顺序完成，没有在中间
更新 policy。16 个 GRPO 组中 14 个有回报差异，非零梯度和有限 KL 说明本轮有有效训练信号，
不再是最小单 chunk 验证中全零 reward/advantage 的情形。
**success 是两轮采样时的 SFT base 结果，不是这次更新后的独立 eval。**
只有一个 scalar 点（原生 TensorBoard step 0，对应保存的 global_step_1），不能判断收敛或提升。

`actor/total_loss` 在原生代码中先除以梯度累积次数再记录，本轮累积 2 次，
因此无额外 loss 项时恰为 `actor/policy_loss` 的一半。不能把它与累积 1 次的旧日志直接比较，
更不能把这部分缩小视为学习变好。本次没有为了对齐日志重写 loss 或训练流程。

GPU CSV 共 1,928 条逐卡样本（241 轮 × 8 卡），每 5 秒采样。
最大 69,407 MiB = **67.7803 GiB**，出现在 GPU 5 的第一轮早期；第二轮和 actor 阶段未超过它。
这不是精确 allocator 峰值；仍需多轮稳定性验证。没有 CUDA OOM 或 NCCL 通信失败。
stderr 仍保留 MSC/Hydra 可选配置源未配置的两段非致命启动 traceback（第 12、23 行），
与 LAM/P2P 修复时的短测一致，随后正常运行；没有屏蔽这些提示。

产物与完整性：

- 原生 checkpoint 实际位于
  `outputs/rlinf/1105305-train/cluster_env64x2_mb8_lam_p2p/checkpoints/global_step_1/actor/`，
  不是 runner log root 下直接的 `checkpoints/`。
  DCP rank 0–7 的 8 个非空 `.distcp` 文件、4,490,458 字节 `.metadata`、
  6,910,740,025 字节 `model_state_dict/full_weights.pt` 均存在。
  合计 20,335,129,612 字节（约 20.34 GB / 18.94 GiB）；尚未重新加载验证。
- 16 个原生 train 视频：`video/train/seed_0` 至 `seed_7` 下各有 `0.mp4`、`1.mp4`，
  对应两轮采样的每-rank 8-env 网格；每个 2560×960、15 fps、121 帧，约 8.07 秒。
  已用本机 PyAV **完整解码全部 16 个视频**；这验证文件完整性，不等于图像清晰度或任务质量评测。
- 本地归档 `logs/cluster_batch64x2_20260911/1105305/`：解析后配置、原生 metrics/TensorBoard、
  worker/Slurm 日志、GPU CSV 和 16 个视频。没有下载大 checkpoint，没有添加源码或 `.diff` 快照。

结论：**64 并行 env × 2 轮采样 / global batch 128 / micro batch 8 在该 8×H100 节点单代验证通过，
可作为 cluster 长训起始配置。** 128 env 同时推理导致的 tokenizer OOM 并未通过本次试验被修复；
本方案是保持每卡 WM batch=8、增加顺序采样量。耗时仍主要集中在 rollout。
后续保持原生每 5 轮 eval/checkpoint，观察独立 eval、组内回报差异及多轮显存；
本轮未启动长训练，未修改正式 YAML 的 max_epochs 或默认 batch。
