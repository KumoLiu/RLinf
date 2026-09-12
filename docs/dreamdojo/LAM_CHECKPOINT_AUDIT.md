# LAM_400k.ckpt 加载审计（2026-09-11）

用户要求回查之前几次实验是否出现 LAM 权重不存在、只打印提示后继续的问题。
结论：**确实出现过，而且最近的 RLinf train/eval 都没有通过该路径加载 LAM 权重。**
此前补齐 `external/lam` 源码只解决 import 失败，不等于加载 `LAM_400k.ckpt`。
最初审计阶段没有修改加载行为。用户随后授权修复，2026-09-11 的实施和验证记录见文末；
以下历史日志结论保留，不把旧实验重新标记为已加载 LAM。

## 日志证据

以下均出现原文：`LAM checkpoint checkpoints/DreamDojo/LAM_400k.ckpt does not exist`。
路径相对 RLinf 根目录；行号以本次审计时的原始文件为准。

| 实验 | 示例日志与行号 |
| --- | --- |
| 早期 RL 运行 09-10 01:11 | `logs/20260910-01:11:20-dreamdojo_trocar_grpo_gr00t_n1d7/run_embodiment.log:1983` |
| 早期 RL 运行 09-10 01:41 | `logs/20260910-01:41:58-dreamdojo_trocar_grpo_gr00t_n1d7/run_embodiment.log:1984` |
| 固定真实动作 WM 检查 | `logs/dreamdojo_review_20260910/wm_gt_v1.console.log:158` |
| 原生长训 native_long_v1 | `logs/dreamdojo_review_20260910/native_long_v1/console.log:1779` |
| noise=0.1 训练 | `logs/20260911-02-50-00-dreamdojo_trocar_noise01/run_embodiment.log:1773` |
| 升级 GR00T 后本地原生 eval | `logs/20260911-gr00t-new-native-eval/worker_logs/EnvGroup/rank_0.log:137` |
| cluster 短训练 1103749 | `logs/cluster_trial_20260911/1103749/worker_logs/EnvGroup/rank_0.log:137` |
| cluster 完整 eval 1104331 | `logs/cluster_eval_20260911/1104331/worker_logs/EnvGroup/rank_0.log:137` |

两个 cluster job 已从远端原始 worker 日志再次复核，不只是读取本地副本。
本地 7 卡新版 eval、native_long_v1、noise01，以及 cluster 两次 8 卡运行，各 Env rank
均出现该缺失提示。最初 cluster job 1103300 则更早因 `external` import 失败退出，
不能把它没有此提示当作 LAM 加载成功。

审计范围为本地 `RLinf/logs` 中的 `.log/.out/.err`：检出 62 个含缺失提示的文件，
没有检出 `Restored LAM`。**文件数不等于实验次数**，包含多 rank、console 和下载的日志副本。
没有日志或日志未覆盖初始化的实验不能仅凭未检出判定成功/失败。

另查本地可见的 `DreamDojo/logs`、`DreamDojo/outputs`：41 个文件含
`Restored LAM from checkpoints/DreamDojo/LAM_400k.ckpt with 0 missing and 0 unexpected keys`，
没有检出上述缺失提示或 Missing/Unexpected LAM keys 提示。例如：

- `DreamDojo/logs/hf_teleop_rollout_posttrain_20260903_163756/01_training.log:47`。
- `DreamDojo/logs/hf_teleop_rollout_posttrain_lora_20260904_040253/01_training.log:47`。
- `DreamDojo/logs/smoke_clip_20260904_155309/state_t4.log:20`。

因此不能笼统说“之前所有 DreamDojo 训练都没加载”。这些日志证明对应旧作业当时恢复成功，
但本次没有审计所有不可见历史作业，也没有据此断言当前 18k checkpoint 的全部训练链路均已核验。

## 为什么没有报错

`DreamDojo/external/lam/model.py:55` 的 `reload_ckpt()`：

- 文件存在：`torch.load(...)["state_dict"]`，然后 `load_state_dict(..., assign=True)`；
  成功后打印 `Restored LAM ...`。没有传 `strict=False`，不能把形状/键不匹配等同于缺文件跳过。
- 文件不存在：只有一个 `print(... does not exist)`，**没有 raise**。
  LAM 模块已被构造，保留初始化参数，而非自动找到另一份 checkpoint。

WM 父类 `text2world_model_rectified_flow.py:186` 将路径硬编码为相对路径
`checkpoints/DreamDojo/LAM_400k.ckpt`。RLinf 创建 WM 时临时切到 DreamDojo 根目录，
所以当前容器实际寻找的是 `/opt/src/DreamDojo/checkpoints/DreamDojo/LAM_400k.ckpt`。
cluster 确实存在权重：

`/lustre/fsw/portfolios/healthcareeng/users/yunl/checkpoints/DreamDojo/LAM_400k.ckpt`

大小 8,518,404,954 字节；但该 checkpoint root 挂载时保留宿主机绝对路径，
**不会自动出现在上面的相对查找路径**。本地 DreamDojo 当前也没有相对 checkpoint 目录。
这与 WM EMA 文件和 reward checkpoint 已正确加载是两件独立的事。

WM 父类的 `state_dict()` / `load_state_dict()`（约 1021 行起）只处理 `net.` / `net_ema.`，
并不会通过已加载的 `model_ema_bf16.pt` 顺便恢复 `self.lam`。

## 对当前 RLinf G1 推理的影响边界

已检查实际源码调用路径：

1. `rlinf/envs/world_model/dreamdojo_adapters.py` 构造 384D 动作，G1 增量位于 `[58:101]`，
   首帧 `[0:29]` 写入归一化 state；最后 32 维 `[352:384]` 保持为零。
2. `DreamDojoEnv._infer_next_chunk_frames()` 直接传入 `action=encoded`、`lam_video=None`。
3. `Video2WorldInference.generate_vid2world()`（约 646 行）中所有 LAM latent 提取及乘入 action
   的语句都是注释。实际进入 `generate_samples_from_batch[_lora]`，然后 conditioner / denoiser，
   不是调用 WM 训练用的 `forward()`。

因此，**当前这条真实关节动作条件推理路径不调用 LAM 网络**；不能因为存在缺失提示就
认定当前 WM 使用随机 LAM 来驱动视频，也没有证据将模糊或低成功率直接归因于它。
这是源码路径与 conditioning 的结论，不是一次“加载/不加载 LAM”的完整 GPU A/B 实验。

不能推广到所有 DreamDojo 用法：WM 父类训练 `forward()`（约 872 行）实际调用 LAM，
将结果乘入 action 最后 32 维；人类视频数据路径会启用这些槽位，缺失权重可能造成无效条件。
当前 G1 dataloader 的对应槽位为零；即便如此，也不能把训练端缺失检查当作可普遍忽略。

建议后续独立改进加载契约：需要 latent-action 的路径应缺失即失败；
真实动作推理应明确声明 LAM 不使用，并避免无声构造未加载模块。
最初审计阶段没有重写 DreamDojo 加载/推理逻辑或改变实验基线。

## 后续授权修复：严格加载与 cluster P2P

改动范围：

- `DreamDojo/external/lam/model.py`：当调用者传入原始默认相对路径时，支持
  `DREAMDOJO_LAM_CHECKPOINT` 覆盖；显式非默认路径优先。路径在构造大模型之前检查，
  文件缺失直接 `FileNotFoundError`，不再留下未加载模块静默继续。
- `torch.load(..., weights_only=True)` 后严格恢复 `state_dict`；缺键、多键、形状错误均失败。
  只有恢复成功才设置 `loaded_checkpoint`，打印绝对路径和 missing/unexpected keys。
  显式 `ckpt_path=None` 仍保留给有意从头训练 LAM 的调用者。
- 不修改 WM 父类、`generate_vid2world()`、G1 的 384D conditioning 或 GR00T 去噪。
  **加载成功不等于当前 G1 RL 视频生成会调用 LAM encoder**，上面的调用路径结论仍成立。
- `RLinf/docker/dreamdojo/run_cluster.slurm`：统一目录中的 LAM 文件必须存在且非空，
  只读同路径挂载并向容器传入环境变量；非 smoke 作业要求严格加载器 API，旧 supplement
  会在启动前失败，不能误用旧版加载器继续训练。
- cluster 默认 `NCCL_P2P_DISABLE=0`，同时覆盖旧镜像内本机包装器的 `:-1` 回退；
  不设置 `NCCL_P2P_LEVEL`。本机包装器仍保留故障规避默认值 1。

部署使用新目录 `code/RLinf-runtime/20260911-lam-strict/external/lam`：
5 个 Python 文件，共 25,908 字节；保留旧 `20260911-lam` 目录，不改共享权重、数据和 v1 sqsh。
LAM 源码是 DreamDojo Git 跟踪文件，其修改可用 `git diff -- external/lam/model.py` 查看。
不把源码快照或 `.diff` 塞进实验日志。

LAM `model.py` SHA256：
`e25bff5039b0dce92cb8e38cadc10a5efebe17bdc8297f6029fa6a4814de1bdc`。
共享 `LAM_400k.ckpt` 再次核对为 8,518,404,954 字节，SHA256：
`d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f`。

验证：

- 全部 DreamDojo CPU 回归：65 passed（11.13 秒），3 项已有 Hydra provider 提示。
  新增 8 项真实加载器的小模型回归；打包测试检查 LAM 缺失/空文件拒绝、只读挂载和 P2P 默认值。
  Ruff lint/format 和 Slurm shell 语法通过。
- 独立 GPU job `1105106`：先恢复实际 LAM 架构和共享权重，使用合成双帧输入检查
  encoder 输出形状和有限性，再以 torchrun 执行八卡 all-reduce、all-gather、各 rank 广播、
  ring send/recv。合成输入仅验证运行正确性，不是机器人任务质量评测。
  原始日志在 `outputs/rlinf/cluster_lam_p2p_20260911/verify-1105106.{out,err}`。
  **已通过**：节点 `pool0-00375`，8×H100 80GB，主作业和所有 step 均 `COMPLETED / 0:0`，
  共 1 分 25 秒。日志第 5 行真实恢复为 `0 missing / 0 unexpected keys`；第 7 行
  `LAM_FORWARD_OK`，输入 `[1,2,240,320,3]`、输出 `[1,1,1,32]`，全部有限，
  latent absolute mean=0.7691378593。注意这是合成视频，不代表任务效果。
  rank 0–7 均有 `NCCL_OK`，每种算子检查 1、65,536、4,194,304 个 float32 元素；
  `NCCL_P2P_DISABLE=0`、`NCCL_P2P_LEVEL=unset`，运行时 NCCL 2.26.2。
  没有做禁用/启用 P2P 的配对性能测试，也未用该日志确认每种消息具体选用的传输路径。
  本地原始日志归档：`logs/cluster_lam_p2p_20260911/1105106/`。

复现独立验证：在 cluster 的 `code/RLinf` 目录，source `docs/dreamdojo/cluster.env.example`，
`unset NCCL_P2P_LEVEL`，再 `sbatch --time=00:15:00 docker/dreamdojo/run_cluster.slurm verify`。
该模式不修改任何模型权重，不运行 policy。`verify_runtime.py` 的 SHA256：
`fbe25142e5d95f03fa32d041b11b2f5a5fa423b227d5fd54401d590636555801`；
部署时 `run_cluster.slurm` SHA256：
`879002541cace7b8f211f6f327f819605080c85683d628912443a1c545b83b37`。

本机同样使用严格加载器，但当前没有找到可用的本地 LAM checkpoint；下一次本机运行前
必须先提供实际 `DREAMDOJO_LAM_CHECKPOINT` 路径，否则会按设计失败。本次没有从 cluster
重复下载 8.5 GB 权重，也没有将“cluster 加载通过”写成本机真实权重测试通过。

### 原生最小集成验证：1105150

独立验证通过后，提交原生 `train` 入口，检查 Ray worker 中的实际加载与训练通信，
不是另写 trainer/evaluator。使用同一严格 LAM supplement 和 P2P=0，提交命令的关键覆盖：

```bash
sbatch --partition=batch --time=00:15:00 --job-name=rlinf-dd-lam-native \
  docker/dreamdojo/run_cluster.slurm train \
  runner.max_epochs=1 runner.val_check_interval=-1 runner.save_interval=-1 \
  runner.resume_dir=null runner.ckpt_path=null \
  runner.logger.experiment_name=cluster_lam_p2p_native_smoke \
  env.train.total_num_envs=64 env.train.rollout_epoch=1 \
  env.train.max_episode_steps=12 env.train.max_steps_per_rollout_epoch=12 \
  actor.global_batch_size=64 actor.micro_batch_size=8
```

SFT base、WM 35 步、policy 4 步、noise=0.1、train 视频开启均不变；
只有 1 个 action chunk（12 条 30-Hz 动作 / env）。完整训练的正式 YAML 不改。
这段窗口不足以完成任务，奖励、success 和梯度大小不能用于判断是否学得起来。
本次不跑完整 eval、不写大型 checkpoint；64 env × 2 轮采样的扩大 batch 实验仍未提交。
日志：`outputs/rlinf/cluster_lam_p2p_20260911/native-1105150.{out,err}`；
原生 runner 输出：`outputs/rlinf/1105150-train`。

**实际结果已通过**：`pool0-00075`，8×H100 80GB；主作业和所有 step 均为
`COMPLETED / 0:0`，5 分 52 秒（容器主 step 5 分 46 秒）。原生 stdout 第 1492–1499 行
覆盖 EnvGroup rank 0–7，每个都从统一绝对路径恢复 LAM，`0 missing / 0 unexpected keys`。
35 步 WM rollout 与 actor 更新阶段正常完成，未出现 LAM 缺失提示、OOM 或异常退出。
`Global Step=1/1`；原生计时 `generate_rollouts=73.220s`、`actor_training=6.045s`、
`sync_weights=15.641s`。不同轨迹长度下不能直接与旧完整 rollout 的耗时比较。
stderr 仍有 MSC/Hydra 可选配置源未配置的两段启动 traceback（第 12、23 行），随后正常继续；
不是 LAM/NCCL 失败。没有为清理日志而隐藏这些既有提示。
仅 0.4 秒的动作窗口内 reward/advantage/grad norm 均为 0，**不视作有效学习或质量提升证明**。
本地完整小型输出归档：`logs/cluster_lam_p2p_20260911/1105150/`，含解析后 config、
metrics、TensorBoard、worker/Slurm 日志和原生 train 短视频，不包含源码快照或大权重。
旧实验日志、旧 supplement、共享 LAM checkpoint 均保留；该修复验证阶段没有启动正式长训或 64×2 扩展实验。
用户随后授权继续的 64×2 完整单代 job `1105305` 已通过，见 [BATCH_SCALING.md](BATCH_SCALING.md)。
