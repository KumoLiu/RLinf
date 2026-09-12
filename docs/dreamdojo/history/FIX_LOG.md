# DreamDojo 修复与实验记录

> 历史修复记录；以下旧命令不再作为启动入口，当前用法见 [README](../README.md)。

日期：2026-09-10。原始审查见 [DREAMDOJO_IMPLEMENTATION_REVIEW.md](DREAMDOJO_IMPLEMENTATION_REVIEW.md)。

最新工作流：用户改为使用七卡原生 train/eval、主 YAML 单入口、每 5 轮 eval / checkpoint。后续启动状态与 55/56 填充统计修正见 [DREAMDOJO_NATIVE_TRAIN_LOG.md](DREAMDOJO_NATIVE_TRAIN_LOG.md)。下文外部 eval 分卡方案为历史方案，不再用于本次长训练。

## 最新状态：性能优化实施与验收

用户随后授权按性能诊断实施优化。已加入固定文本缓存、零 guidance 单分支、rollout 边界卸载，并修复 trial 配置继承、TensorBoard 路径及独立诊断导入。完整修改和失败／重试过程统一记录于 [DREAMDOJO_OPTIMIZATION_LOG.md](DREAMDOJO_OPTIMIZATION_LOG.md)。固定 batch8、35 步 WM 测得 64.28 → 25.75 秒／chunk；完整 batch3 策略闭环 18.85 → 10.18 秒／chunk，数值对照均最大误差 0。五卡两轮训练正常结束，耗时 13.33／12.14 分钟，两轮指标均有限，checkpoint 已保存。优化开关已启用，未降低 35 步，未宣称 RL 收益。没有自动启动 20 轮 trial，也没有使用旧 step20。

## 历史状态：完整基线结束，性能诊断完成

2026-09-10 11:32 UTC 更新：两个 30k 分片已成功补跑，完整 55 条基线汇总于 `logs/dreamdojo_review_20260910/sft_full55_35steps_seed0_retry01/results.json`。总体 reward-model success_once 为 15/55（27.27%），teleop 12/25、30k 3/17、10k 0/13。下文“38/55、待补跑”等为此前时点的历史记录，不是当前状态。

用户授权的 20 轮 trial 尚未启动；随后用户要求先分析耗时。本轮只读核对日志／源码、执行无模型加载的 guidance 控制流 probe，并新增 [DREAMDOJO_PERFORMANCE_REVIEW.md](DREAMDOJO_PERFORMANCE_REVIEW.md)。没有实施性能优化、改变去噪步数或启动 GPU 训练。主要发现：固定文本 embedding 重复计算和模型反复搬运；guidance=0 仍计算两路 DiT；两卡完整评估需要考虑固定分片排队。报告区分旧 5 步 smoke、35 步独立评估及未执行的五卡 trial，包含计时证据和验证顺序。

前一轮 trial 脚本／配置仍是未启动草稿，测试曾中断，不能认定验收通过。本轮发现健康监控 TensorBoard 路径不匹配，已登记为启动前必须修正和验证的事项，未在性能诊断中静默修改。

## 追溯方式

- 修复前代码、未提交 Git 差异和 SHA256 清单：`logs/dreamdojo_review_20260910/before/`。
- 修复后代码快照及仅相对 before 的补丁：`logs/dreamdojo_review_20260910/after/`，其中 `changes_since_before.patch` 不混入 before 已有的用户改动。
- 可复现诊断入口：`toolkits/world_model/dreamdojo_validation.py`。
- 每个诊断输出目录保存运行参数、源码快照与结果。原始模型和数据不做覆盖。
- 当前修复尚未提交 Git；用户原有未提交改动保留，before 快照用于区别原有实现与本次修改。

## 实现变更

历史修复说明：Claude 提到的“absolute arm state 被错误重复累加”在本次审查开始前已经修复。before 快照中的 `update_g1_dex3_state()` 已直接返回 `actions.clone()`，本次没有把它当作现存问题，也没有重复实施该修复。V2 指的是“动作目标代替实际测量状态”的剩余近似，与重复累加是两个问题。base／step20 对照会使用同一套修正后的环境；旧 step20 权重可能保留旧环境下训练的影响。

| 审查项 | 修改 | 验证状态 |
| --- | --- | --- |
| F1 | G1 物理目标不再按归一化边界裁剪；recipe 关闭解码后额外动作噪声，保留 flow-SDE | 回归测试通过；两轮 GRPO smoke 通过 |
| F2 | min-max 归一化取消额外 clamp，与已有 checkpoint 的训练约定一致 | 2,486 窗口 parity 最大差 0 |
| F3 | 从 observation.state 统计量归一化历史状态，填入第一行前 29 维 | 完整 384D parity 通过 |
| F4 | 12 个 30 Hz 命令消费 6 个 15 Hz WM 帧；末尾动作只填充未执行的未来 | 单测与 GT 回放通过；未来保持仍是近似 |
| F4 reset | 条件图像从 I2 开始，WM baseline 为 a0、s0；策略 proprio 使用 s2 | 真实数据 reset 断言通过 |
| V1 | 默认恢复 35 步，单独比较 5／35 步 | GT 重建未见优势；策略闭环敏感，保留 35 步质量参考 |
| V2 | 保留完美跟踪目标的代理状态假设，并分别维护当前状态和 WM 历史状态 | 代理闭环已跑；近似本身的影响未隔离 |
| 日志 | 增加三个阶段完成率、当前 chunk 三头概率峰值和 episode 秒数 | 两轮训练／评估日志已验证 |

`max_episode_steps` 改为 240 个 30 Hz 控制命令，保持约 8 秒的环境时间。每个 WM 帧的奖励放在对应两个控制命令的第二个位置。`ignore_terminations` 现在影响返回的终止标记，success 指标仍独立记录。

当前明确支持单条件图像、KIR 关闭、auto_reset 关闭。原实现的其他组合未完成契约，改为显式报错，避免静默使用不完整行为。

## 时间与状态约定

以当前图像 I[t] 为准：

- 策略输入状态为 s[t]；执行 a[t] 到 a[t+11]。
- WM 编码使用 a[t-2] 作 baseline，取 a[t], a[t+2], ..., a[t+10]；额外未来用末尾已知命令保持填充。
- 生成 12 个未来 WM 帧，只消费 I[t+2] 到 I[t+12] 的前 6 帧。
- 下一个 WM baseline 是 a[t+10]；baseline 状态 s[t+10] 在当前代理假设下取 a[t+9]；当前状态 s[t+12] 取 a[t+11]。

未来保持填充是有限策略 horizon 下的近似。虽然未消费对应未来帧，但网络可利用整段条件，必须通过 GT 回放实验测量其影响。proprio 仍不是 world model 预测出的真实状态。

## 实验进度

1. 修复前源码快照：已完成，36 个源码／配置／文档文件。
2. 单元测试：12 项通过，最终日志 `logs/dreamdojo_review_20260910/tests_final.log`（12.69 秒）；包含物理目标不误裁剪、超统计范围数据、状态槽、时间轴、生成帧截取、历史更新和奖励位置。真实数据 parity 已通过：225 条 teleop 数据的 2,486 个窗口，完整 384 维条件最大绝对差为 0，reset 图像／baseline 断言通过。结果见 `logs/dreamdojo_review_20260910/parity/results.json`。新增诊断脚本及修改的生产代码 Ruff 检查通过，`git diff --check` 通过。
3. GT actions 的 5／35 步、teacher forcing／闭环对照：已完成，使用留出集 episode 0、1、2，各 20 个控制 chunk（8 秒），耗时 884 秒；输出目录 `logs/dreamdojo_review_20260910/wm_gt_v1/`。
4. 真实／生成视频奖励检查：真实视频和四组生成视频均为 return [3,3,3]；抽帧确认 GT 闭环 episode 0 存在拿起、转移和右侧放置运动，仍有形变。完整概率曲线与动作图位于 `logs/dreamdojo_review_20260910/wm_report/`。
5. SFT／旧 step20 策略闭环对照：已完成，目录 `logs/dreamdojo_review_20260910/policy_v1/`。base return [1,1,3]，step20 return [3,3,3]；即本组三起点上的 reward-model success_once 分别为 1/3、3/3。抽帧核对 base episode 2 和 step20 episode 0 有物体转移和右侧放置。使用 35 步 WM、相同 seed 0、相同起点和代理状态；不是完整验证集或实机结果。曲线和时刻汇总见 `logs/dreamdojo_review_20260910/policy_report/`。
6. 小规模 GRPO：已完成，退出码 0、含初始化总耗时 2,157 秒。独立目录 `logs/dreamdojo_review_20260910/grpo_smoke_v1/`，GPU 2、8 个训练环境、group size 4、2 轮 runner 迭代、5 次 WM 去噪、8 秒任务、3 个评估起点。用于验证训练链路，不用于估计最终性能；从 SFT 权重重新开始，不续训旧 step20。所有 GPU 实验均已退出。

第一轮 GRPO（runner step 1）：训练 return 1.75、success_once 3/8，picked 7/8、handed 4/8；优势范围约 [-1.162,1.162]、grad_norm 0.984、approx_kl 8.48e-4，checkpoint 已保存。评估 return 1.333、success_once 0/3。链路存在有效学习信号，但本轮没有证明泛化收益；5 步 smoke 评估不能直接与 35 步策略诊断作性能比较。

第二轮 GRPO（runner step 2）：训练 return 1.0、success_once 0/8，picked 6/8、handed 2/8；优势范围约 [-0.866,0.866]、grad_norm 1.068、approx_kl 3.45e-4。评估 return 1.0、success_once 0/3。两轮评估均没有完成任务，因此不能把“接口修复／训练链路通过”写成“RL 已经提升策略”。训练每轮重新抽取起点，训练成功率也不是同起点前后对照。

精确 TensorBoard 标量和新权重路径导出于 [grpo_report/results.json](logs/dreamdojo_review_20260910/grpo_report/results.json)。TensorBoard step 0／1 对应 runner 与 checkpoint step 1／2。新权重仅用于本次 smoke 记录，未替换原策略。

## 当前结论与下一步

### 完整 SFT 基线：2026-09-10 续跑

用户确认继续验证后，已启动原始 SFT 的完整 55 条验证。原输出目录为 `logs/dreamdojo_review_20260910/sft_full55_35steps_seed0/`。38／55 条已完成；用户重启后再次确认启动，剩余 17 条已于 10:11:57 UTC 在独立目录 `sft_full55_35steps_seed0_retry01/` 补跑。原评测进程已于 09:09 UTC 结束，旧错误和 38 条有效结果均保留。没有启动长训，旧 step20 不参与本轮。

- 数据覆盖：teleop validation 25 条、30k rollout validation 17 条、10k rollout validation 13 条；按 dataset＋episode 唯一键检查，无混合随机采样或补齐到 56 条。
- 协议：seed 0、WM 35 次去噪、20 个控制 chunk／8 秒、I2 起点、动作目标代理状态。调用既有 GR00T → RLinf DreamDojoEnv 路径，不运行 Ray 或优化器。
- 固定分片：GPU 0／1／2 的 teleop 分别为 0–8／9–16／17–24；GPU 3／4 的 30k 为 0–8／9–16；GPU 5／6 的 10k 为 0–6／7–12。每条只评估一次。
- 每个分片独立以 seed 0 初始化；批大小和 batch 内次序影响随机数消耗，因此未来 checkpoint 对照必须复用 `plan.json` 的分片和顺序，不能直接与此前 batch=3 的小样本结果逐条归因。
- 新增 `toolkits/world_model/dreamdojo_full_eval.py` 负责分片、子进程管理与完整性汇总；现有策略推理和模型配置不变。新增 7 项覆盖／去重测试，加已有 12 项共 19 项通过（10.23 秒），Ruff 与 `git diff --check` 通过。
- 数据集中原始 rollout 的 success/fail 标签描述旧实机轨迹，不是此次新生成轨迹的正确答案；最终报告仍是 reward-model 成功判定。

本轮不是固定真实动作回放：SFT base 本身是策略，每个 chunk 由它预测动作，再由 WM 生成后续观测。eval 数据只初始化 I2／s2 和所需历史 a0／s0，之后不把真实后续图像、动作或状态输入闭环。GR00T、WM、奖励模型权重均固定；目标是建立修复后系统的 SFT 基线，供未来 RL checkpoint 在同协议下比较，不代表已在验证 RL 收益或实机成功率。

运行异常与恢复记录：两个 30k 分片（`rollout_30k_00`、`rollout_30k_01`）遇到 `CUDA error: unknown error`，分别在首个 chunk 的 VAE 解码、第二个 chunk 的 DiT 推理处中断，不是已确认的 OOM。随后 `nvidia-smi` 从启动时 8 个设备变为 7 个设备；未获得内核 Xid 日志，不能确定根因。其余五个分片继续运行，未重置驱动或修改 GPU 配置。运行错误不作为策略失败计入分母。

为保证恢复可追溯，完整评估工具增加 `--retry-plan`／`--retry SHARD=GPU_UUID`：仅允许补跑已失败的分片，保留其种子、批大小和 episode 次序，使用独立新目录保存补跑结果；已完成分片只读复用，原计划和错误日志保留，最终每条记录附结果来源路径。GPU 重新枚举后改用稳定 UUID 绑定设备。汇总仍严格要求 55 条有效结果，缺失不补零。

09:10–09:13 UTC 的进一步只读检查：原 GPU 1、2 对应 UUID 的新进程，以及 `CUDA_VISIBLE_DEVICES=0` 的新进程，均在极小矩阵计算前初始化 CUDA 失败。`nvidia-smi -q` 对全部 7 个可枚举 GPU 报告 `GPU Recovery Action: Reboot`。GPU 计算进程已全部退出；仍存在 Xorg 图形进程。没有执行 GPU reset、服务重启或机器重启。恢复机器运行环境需要用户／管理员安排，之后才能补齐缺失分片；不能将故障归因于具体模型代码或已证实的显存不足。

已完成部分的结果（各阶段为累计完成数）：

| 验证起点来源 | 有效／计划条数 | picked | handed | placed／success_once | 平均 return |
| --- | ---: | ---: | ---: | ---: | ---: |
| teleop validation | 25／25 | 24／25 | 13／25 | 12／25 = 48% | 1.96 |
| 10k rollout validation | 13／13 | 0／13 | 0／13 | 0／13 = 0% | 0 |
| 30k rollout validation | 0／17 | 待补跑 | 待补跑 | 待补跑 | 待补跑 |

不发布 55 条的总体成功率。机器可读部分汇总为 `sft_full55_35steps_seed0/partial_summary.json`；五个分片的逐帧三阶段概率和逐关节动作图在 `sft_full55_35steps_seed0_partial_report/`，由保存的 trace 在 CPU 上生成。这里只评价 8 秒生成轨迹，不是原始数据集成功标签或实机成功率。

抽帧检查了 `teleop_00/base/episode_000.mp4` 和 `rollout_10k_00/base/episode_000.mp4`：前者可见物体从左侧转移到右侧，后者未见完成转移，生成画面仍有形变。这不足以确认全部奖励判定，也不足以把数据源间差异归因于单一原因。下一步应补齐 30k 基线，并对 rollout 起点增加真实动作回放／奖励人工核对，再决定短训练；暂不启动长训。

GPU 暂停后补做两份 rollout validation 的 CPU parity：10k 数据 145 个窗口、30k 数据 146 个窗口均零失败，完整 384D conditioning 最大绝对误差均为 0，episode 0 的 I2／a0／s0／s2 reset 断言通过。结果分别在 `parity_rollout_10k_val/results.json` 与 `parity_rollout_30k_val/results.json`。这证明这些真实输入窗口的 bridge 变换与原生处理一致，不证明 rollout 起点的 WM 质量、策略能力或目标状态代理有效。

恢复工具新增 3 项测试，加前述 19 项共 22 项通过（7.73 秒）；本轮 CPU 测试出现 CUDA 初始化警告，与同机 GPU 故障一致，但测试退出码为 0。Ruff lint／format 检查通过。

恢复后的补跑命令（已于 10:11:57 UTC 执行；如再次复现必须改用不存在的新 `--output`，并重新确认 CUDA 可用和 UUID 空闲）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python toolkits/world_model/dreamdojo_full_eval.py \
  --retry-plan logs/dreamdojo_review_20260910/sft_full55_35steps_seed0/plan.json \
  --retry rollout_30k_00=GPU-3b9e2a3e-1e46-2d99-9981-ed43b55d63a9 \
          rollout_30k_01=GPU-170ffc1d-37ce-364a-64ba-23da29ee5466 \
  --output logs/dreamdojo_review_20260910/sft_full55_35steps_seed0_retry01
```

补跑读取原 plan 中的 seed 0、35 次去噪、20 chunks、9／8 batch 和 episode 顺序，不重跑五个正常完成的分片。补跑完成后在新目录汇总 55 条，并记录每条结果的来源。

重新启动时，GPU 0／1／2／3／5／6／7 全部通过新进程中的 CUDA 64×64 矩阵计算，整个检查进程的可见列表排除了 GPU 4 的 UUID。这是可用性检查，不是长时压力测试。仅避开 GPU 4，其余七张均保留为后续候选；本次只需要 GPU 1／2 承担两个原始分片，不通过拆细 batch 来占满七卡，以保持既定随机采样协议。实际进程为 `rollout_30k_00` PID 12997（GPU 1，episodes 0–8）和 `rollout_30k_01` PID 12998（GPU 2，episodes 9–16），两者绑定稳定 UUID。启动记录在补跑目录 `RESTART.md`，运行状态见该目录 `process_status.json`。

### 重启后定位与避卡：2026-09-10 09:55 UTC 起

用户重启后，8 张 GPU 均重新枚举、恢复动作均为 `None`。只读管理员日志现已可访问：主机 `smc521ge-0004` 于 08:51:18 UTC 记录 `AD:00.0` 的 PCIe `Link Down`／`Card not present`，随后为 Xid 79：`GPU has fallen off the bus`。该卡是 **GPU 4（零起始编号，列表第 5 张）**，UUID `GPU-77264942-9d19-703f-ff04-ee8e38ce7cb1`，序列号 `1333324150150`。全部卡随后报告 Xid 154、要求整机重启；不能把其他卡的恢复标记或 GPU 3 的进程退出也认定为独立硬件故障。

这确认了掉线设备，尚不能区分显卡本体、PCIe 连接、供电等根因。按用户要求，后续实验选卡排除该 UUID；本次补跑继续固定为 GPU 1／2 的 UUID，两卡新进程的 CUDA 64×64 矩阵计算均通过（`available True`，`matmul_mean 64.0`）。没有对 GPU 4 跑计算，也没有系统级禁用设备。原始错误日志保留，新增内核证据位于 `sft_full55_35steps_seed0/kernel_fault_20260910_excerpt.log`，详细记录在同目录 `GPU_DIAGNOSTICS.md`。

完整重新评测的复现命令（不是剩余 17 条的补跑；`--output` 必须为新目录）：当前映射下用 GPU 7 替代 GPU 4，保持七个分片的 batch 和 episode 次序不变。执行前仍需核对 UUID 与空闲状态；该命令本轮未运行。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python toolkits/world_model/dreamdojo_full_eval.py \
  --gpus 0 1 2 3 5 6 7 --seed 0 --steps 35 --chunks 20 \
  --output logs/dreamdojo_repeat/sft_full55_35steps_seed0
```

### 前一轮结论与初始化决定

已完成 F1–F4 的接口修复、CPU 回归、原生 384D parity、真实动作 2×2 回放、同时间轴奖励核对、base／step20 在 5／35 步的闭环对照，以及两轮 GRPO smoke。源码、配置、标量、动作／概率轨迹和视频均已归档。

初始化决定（2026-09-10，用户确认）：旧 step20 在接口修正前训练，不再作为后续训练初始化或评估候选。后续统一从原始 SFT checkpoint 开始，不恢复旧 step20 的权重、优化器或训练状态。已有 step20 结果仅作历史诊断记录，文件保留用于追溯，未执行删除。这一决定替代此前“扩大 base／step20 对照后再选择初始化”的建议。

仍需验证的是训练收益，而不是继续假定 pipeline 正常就应该有高成功率。下一轮先在统一的 35 步评估协议下建立原始 SFT 的完整验证基线，再从 SFT 启动小规模 35 步训练，比较新 checkpoint 与相同起点、种子下的 SFT 基线。35 步暂作质量参考，不把旧 step20 的三起点好结果当作新训练能获益的证据。还应单独检验状态代理、未来动作保持填充，以及失败生成视频上的奖励误报。完整 55 条验证集、多种子评估、实机评估和逐项修复消融本次均未完成。

## 策略闭环的 5／35 步对照

| 策略 | WM 去噪次数 | episode 0 / 1 / 2 的 return | reward-model success_once |
| --- | ---: | --- | ---: |
| base | 5 | [1,3,0] | 1/3 |
| base | 35 | [1,1,3] | 1/3 |
| 旧 step20 | 5 | [2,2,2] | 0/3 |
| 旧 step20 | 35 | [3,3,3] | 3/3 |

除去噪次数外，使用相同 checkpoint、seed 0、起点、状态代理和 8 秒时长。5 步结果在 `policy_5steps/`（耗时 256 秒），35 步结果在 `policy_v1/`，全部曲线汇总在 `policy_steps_report/`，均位于 `logs/dreamdojo_review_20260910/`。

这说明固定动作回放正常，并不保证策略闭环对采样步数不敏感：图像的细微差异可改变后续动作。保留 35 步为当前默认质量参考；5 步只用于已明确标注的快速训练链路实验。尚不能依据三个起点认定 35 步普遍更好、旧 step20 已泛化，或在实机上达到同样成功率。

## GT 回放结果与解释

| 输入方式 | 去噪次数 | 三条轨迹 reward-model return | 平均 PSNR（dB） | 平均像素 MSE |
| --- | ---: | --- | ---: | ---: |
| 真实图像逐 chunk 重置 | 5 | [3,3,3] | 22.224 | 393.26 |
| 连续生成 | 5 | [3,3,3] | 17.867 | 1084.96 |
| 真实图像逐 chunk 重置 | 35 | [3,3,3] | 21.856 | 427.91 |
| 连续生成 | 35 | [3,3,3] | 17.542 | 1158.34 |

PSNR／MSE 比较的是时间对齐的真实未来帧，排除初始条件图像及真实视频结束后的补帧。表中先对各 episode 求指标，再对三条等权平均。它们测量像素重建误差，不等价于感知清晰度。

当前证据支持：修正接口后，真实动作可以驱动 WM 产生有效任务运动；闭环误差明显大于 teacher forcing。此小样本没有证明 35 步比 5 步更好，也没有排除更广泛的 reward 域偏移。reward-model 判定不是实机成功率。四个修复同时应用，不能据此归因每一项各自贡献。

补充同时间轴的真实视频概率对照：`real_reward_aligned/` 从 I4 开始评分，与 WM 消费的未来帧一致；真实视频结束后保持最后一帧，均执行到 8 秒。三条真实轨迹仍为 [3,3,3]，四组生成轨迹的阶段完成判定时刻与对应真实轨迹最大相差约 0.133 秒。该结果支持这三个正样本上未见明显 reward 域偏移，但不覆盖失败样本上的误报率。

可直接查看：[真实／生成概率叠加图](logs/dreamdojo_review_20260910/reward_report/recorded_vs_generated_probabilities.png)、[真实动作闭环视频（5 步，episode 0）](logs/dreamdojo_review_20260910/wm_gt_v1/gt_closed_loop_5steps/episode_000.mp4)、[base 成功样例（episode 2）](logs/dreamdojo_review_20260910/policy_v1/base/episode_002.mp4)、[step20 样例（episode 0）](logs/dreamdojo_review_20260910/policy_v1/step20/episode_000.mp4)。

回放给了真实历史 proprio；策略闭环仍使用动作目标近似状态，因此两者差距还可能来自状态代理。对所有验证轨迹、多个随机种子和实机评估的结论尚未建立。

## 复现入口

在 `/localhome/local-yunl/RLinf` 下执行。每个 `--output` 必须是新目录，脚本会拒绝覆盖既有结果。GPU 编号应先确认空闲；本次实际运行的完整参数已存入各目录的 `manifest.json`。

```bash
# 单元测试：不使用 GPU。
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests/unit_tests/test_dreamdojo_adapters.py

# 原生动作／状态变换一致性。
CUDA_VISIBLE_DEVICES='' .venv/bin/python toolkits/world_model/dreamdojo_validation.py parity \
  --dataset /localhome/local-yunl/data/pick_trocar_teleop_success_train \
  --output logs/dreamdojo_repeat/parity

# 真实动作：四组生成对照。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python toolkits/world_model/dreamdojo_validation.py world_model \
  --episodes 0 1 2 --chunks 20 --steps 5 35 --output logs/dreamdojo_repeat/wm

# 历史策略对照复现：仅用于追溯，不属于下一轮训练／评估计划。
CUDA_VISIBLE_DEVICES=1 .venv/bin/python toolkits/world_model/dreamdojo_validation.py policy \
  --episodes 0 1 2 --chunks 20 --steps 35 --checkpoints base \
  /localhome/local-yunl/RLinf/logs/20260910-01:41:58-dreamdojo_trocar_grpo_gr00t_n1d7/dreamdojo_trocar_grpo_gr00t_n1d7/checkpoints/global_step_20/actor/model_state_dict/full_weights.pt \
  --output logs/dreamdojo_repeat/policy

# 从保存的 trace.npz 生成图表，不重新跑模型。
.venv/bin/python toolkits/world_model/dreamdojo_validation.py report \
  --inputs logs/dreamdojo_repeat/wm logs/dreamdojo_repeat/policy \
  --output logs/dreamdojo_repeat/report
```

GPU 运行沿用原启动脚本的线程池限制（OMP／MKL／OPENBLAS／TF 均为 1）；本次设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 使用已经存在的本地权重。短训练的解析后完整配置在 `grpo_smoke_v1/train_config.yaml`，其中“2 步”指两轮 runner 迭代，每轮内部可含多次 optimizer step，不代表只做两次梯度更新。

旧 step20 权重未修改或删除，但按用户决定不再使用；后续从原始 SFT 重新建立基线和训练。本次小实验不足以证明实机收益。

## 实验术语

- parity：同一真实动作／状态经 DreamDojo 原生数据处理与 RLinf bridge 后，比较完整 384 维模型条件。通过只证明输入变换一致，不证明视频质量或任务成功。
- 固定真实动作：用成功演示的 recorded actions 替代 GR00T 输出；本轮也给 WM 提供真实历史状态，隔离策略和 proprio 近似的影响。
- 5／35 步：每段视频生成的去噪迭代次数，不是机器人动作数或 RL 训练步数。
- teacher forcing：每个 chunk 重置为对应时刻的真实图像；closed-loop：仅首图真实，之后接续模型生成图像。
- 三阶段概率：奖励模型对拿起／交接／放置的判断，需要与视频核对，不能独立当作真实任务成功率。
