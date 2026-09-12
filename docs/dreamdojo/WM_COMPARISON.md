# 新旧 World Model 对比：scratch_lr3e-4（2026-09-11）

已在 cluster 找到 `lora_r32_scratch_lr3e-4_18k` 的最终 EMA 权重。
只新增统一 checkpoint 路径，不覆盖旧权重、不改默认 YAML、不复制整份训练配方。
现有原生 Slurm 入口已经支持通过 checkpoint 环境变量和 Hydra 参数切换。

## 权重与配套配置

统一根目录：`/lustre/fsw/portfolios/healthcareeng/users/yunl/checkpoints/DreamDojo/`。

| 项目 | 原 WM | 新 WM |
| --- | --- | --- |
| 子目录 | `lora_r32_lr3e-4_r64_18k` | `lora_r32_scratch_lr3e-4_18k` |
| 文件（两者相同后缀） | `checkpoints/iter_000018000/model_ema_bf16.pt` | `checkpoints/iter_000018000/model_ema_bf16.pt` |
| LoRA rank / alpha | 64 / 64 | 32 / 32 |
| DreamDojo experiment | `dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora_lr3e-4_r64` | `dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora` |
| 文件大小 | 4,486,867,523 bytes | 4,395,116,963 bytes |

新权重原始位置：
`/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/train/dreamdojo/hf_teleop_rollout_posttrain_lora/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt`。

已在统一目录建立硬链接，原始文件和新路径共用同一个 inode，没有传输或复制 4.4 GB 权重。
旧 rank 64 权重保留原位置。硬链接不是独立备份，不能沿其中一个路径覆盖写入；RLinf 只读挂载这些输入。

新权重 SHA256：`2ea3050704a31e40e3f90cd9457676d809622d62dcf46bc010b03603e66fe4b4`。
旧权重已记录 SHA256：`267f2209e3e58fec5a153d8a92984e05a60005f74661e2e6555e40e1b66147b0`。
新 run 的 `latest_checkpoint.txt` 为 `iter_000018000`，文件大小稳定，PyTorch ZIP 中有 1263 个条目。
这是文件格式/存在性检查，不代表已经完成 GPU 加载或视频质量验证。

`scratch` 的实际含义：从官方 `2B_G1_post-train/iter_000050000` 权重开始，
以 LoRA rank 32、lr=3e-4 适配 18,000 steps，没有接着旧 iter-3000 适配 checkpoint 训练；
**不是整个 2B 世界模型随机初始化训练**。该 lr 是 WM 训练时的学习率，不是 RL actor 学习率。
依据：DreamDojo `scripts/submit_lora_experiment.sh` 及该 run 的 `config.yaml`、`launch_info.yaml`。

## 为什么权重路径与 experiment 必须一起换

RLinf 的 `env.train.experiment_name` / `env.eval.experiment_name` 决定 WM 网络及 LoRA 结构，
`DREAMDOJO_WM_CHECKPOINT` 只决定加载哪个文件。只换路径而保留旧 rank 64 配置会发生形状不匹配。
新配置已存在于当前 v1 镜像的构建快照中，无需重新打包 Docker。

两者推理接口兼容：384D action conditioning、`state_t=4`、13 帧/12 actions chunk、
相同 G1 action bridge 和归一化文件；选择新 WM 不应更改这些设置。
scratch 配套基础 experiment 内的 optimizer.lr 默认值不影响推理：RL 环境不训练 WM optimizer，
加载的是上述已经以 3e-4 训练好的完整 EMA 权重。

## 新 WM 的原生 eval / train 命令

以下是原生入口的命令模板。本轮六组提交记录见文末；没有额外提交独立 base eval。
在 cluster 的 RLinf 目录执行，先 source site 设置，**再**选择 WM，避免被 site 的旧默认值覆盖：

```bash
cd /lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf
source docs/dreamdojo/cluster.env.example
export DREAMDOJO_WM_CHECKPOINT="${RLINF_CHECKPOINT_ROOT}/DreamDojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt"
WM_EXPERIMENT=dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora

# 首先评测 SFT base 在新 WM 中的完整 55 个固定案例，保存原生 eval 视频。
sbatch --job-name=dd-wm-r32-base-eval docker/dreamdojo/run_cluster.slurm eval \
  runner.logger.experiment_name=wm_scratch_lr3e4_r32_base_eval \
  "env.train.experiment_name=${WM_EXPERIMENT}" \
  "env.eval.experiment_name=${WM_EXPERIMENT}"

# 以下是独立的 RL 长训示例，不是上面 eval 完成后会自动启动的任务。
# 确认对比矩阵及验证结果后再提交。
export RLINF_CHAIN_SOURCE_ROOT=/lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf-runtime/20260911-chain-v1
export CHAIN_TIMEOUT=3.9h
export MAX_RUNS=18  # 本轮链长上限，不保证覆盖完整 1000 代
sbatch --job-name=dd-wm-r32-noise01 docker/dreamdojo/train_chain.slurm \
  runner.logger.experiment_name=wm_scratch_lr3e4_r32_noise01 \
  "env.train.experiment_name=${WM_EXPERIMENT}" \
  "env.eval.experiment_name=${WM_EXPERIMENT}" \
  env.train.total_num_envs=64 env.train.rollout_epoch=2 \
  actor.global_batch_size=128 actor.micro_batch_size=8 \
  actor.model.rl_head_config.noise_level=0.1
```

原 WM 对照组只更改这三项选择（另取独立的实验名）：

```bash
export DREAMDOJO_WM_CHECKPOINT="${RLINF_CHECKPOINT_ROOT}/DreamDojo/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt"
WM_EXPERIMENT=dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora_lr3e-4_r64
# 重用上面的 eval/train 命令，分别使用带 r64 的 runner.logger.experiment_name。
```

Slurm 会保留 checkpoint 环境变量和两个 experiment 参数，自动续跑不会切回默认的旧 WM。
不要在已有 chain 中途更换 WM；新旧组分别从相同 SFT base 初始化，新建 chain，
不要加载修复前 step20 或续跑机制的 12-action 测试 checkpoint。

## 六组主实验设计（2026-09-11）

用户选择两个 WM × Flow-SDE noise_level=0.1 / 0.3 / 0.5。
首批提交六组主实验，global batch=128；随后用户授权在 scratch/noise=0.3 下追加三组 LR/batch 对照，见文末。
其他备用组合没有提交。实际主实验名额外包含 _gb128。

| 实验名（首个训练 seed） | 训练 WM | noise_level | action_noise_scale |
| --- | --- | --- | --- |
| `wm_r64_n01_s1234` | 原 rank 64 | 0.1 | 0.0 |
| `wm_r64_n03_s1234` | 原 rank 64 | 0.3 | 0.0 |
| `wm_r64_n05_s1234` | 原 rank 64 | 0.5 | 0.0 |
| `wm_scratch_r32_n01_s1234` | scratch rank 32 | 0.1 | 0.0 |
| `wm_scratch_r32_n03_s1234` | scratch rank 32 | 0.3 | 0.0 |
| `wm_scratch_r32_n05_s1234` | scratch rank 32 | 0.5 | 0.0 |

这比较的是两个**完整 WM checkpoint 方案**，不能单独归因于 LoRA rank；它们的初始化/训练历史也不同。
所有组从同一个 GR00T SFT base 开始，不能从已有 RL ckpt 接着跑。

### 固定项

- 使用已经验证过的 cluster 配方：8 卡、`env.train.total_num_envs=64`、
  `env.train.rollout_epoch=2`、`actor.global_batch_size=128`、`actor.micro_batch_size=8`。
  这是 CLI 覆盖；本机 YAML 默认的 56 env / 1 rollout / global 56 / micro 2 不代表此矩阵。
- `group_size=8`，每代 128 条轨迹、16 个 GRPO 组；每轨迹 240 个 30-Hz actions、20 个 policy chunks。
- WM train/eval 均 35 步；GR00T 去噪 4 步。`noise_anneal=false`，不同时改变推理步数或噪声 schedule。
- actor lr=5e-6，update_epoch 默认 1；保留当前优化器、PPO clipping、三阶段奖励与 `filter_rewards=false`。
- 相同数据版本、reset 混合比例 `[0.34,0.33,0.33]`、初始帧重置、action statistics、SFT processor、
  reward ckpt、LAM 严格加载、Docker/runtime 源码版本；cluster P2P 开启，不沿用本机 GPU 4 故障排除。
- 第一轮六组使用相同 `actor.seed=1234`、`env.train.seed=0`、`env.eval.seed=0`，保持 batch/worker 布局不变。
- 原生 train/eval、val/save 每 5 代、上限仍为 1000 代；按 20 / 50 代检查走势，不设 20 代硬停止。
  每个实验独立 chain/log，保留原生视频；不额外复制 source 或 `.diff`。

沿用上方启动命令，只选择 WM 的 checkpoint + train/eval experiment 名，再覆盖表中的
`actor.model.rl_head_config.noise_level`、`runner.logger.experiment_name`。
建议提交时显式写 `actor.model.rl_head_config.action_noise_scale=0.0`，便于从保存的 config 审计。

### 两种 noise 的准确含义

`actor.model.rl_head_config.noise_level` 控制 GR00T **归一化 action latent 去噪过程**中的 Flow-SDE 探索。
当前 `joint_logprob=false`：每次调用随机选择一个去噪 index，在该步使用随机转移，其他步用确定性转移。
同一 batch 共享被选中的 index，但每个元素有各自的高斯噪声；该参数也参与 SDE 转移均值的修正。
它不是 DreamDojo 视频像素噪声，不改变 WM denoising steps，也不是物理关节偏移的单位。

`actor.model.rl_head_config.action_noise_scale` 是另一个独立开关：在 processor 解码并转换成
28-D **物理绝对关节目标之后**，对各 env、各时间步、各关节叠加同一标量标准差的独立高斯噪声。
G1 分支不会在这里执行 LIBERO 用的 `[-1,1]` clamp；当前实现也不区分手臂/手指或做时间平滑。
这层噪声在 RL denoising chains / log-probabilities 计算之后添加，不是另一份可学习的 Flow-SDE 转移。
不要据此宣称算法必然错误，但它应被视为额外的执行扰动，不能与 noise_level 混为一谈。

**主矩阵不增大 physical action noise，固定为 0。** 原因是此时要识别 WM 与策略探索的影响；
额外逐帧关节抖动可能使动作偏离 WM 的训练分布，并增加视觉/关节状态不一致。
若主实验选出候选后仍需要测试，再固定 WM 和 noise_level，单独比较 action_noise_scale=0 / 0.01 / 0.02。
这些数值是原始动作单位，不是 1% / 2%；先核对各关节单位和演示动作变化尺度，再通过短 rollout 检查越界、
速度/抖动及视频。它是低优先级的鲁棒性实验，不预设增加后一定更好，更不直接跳到 0.1 / 0.3 / 0.5。
该建议只针对 WM 实验，不能未经安全验证直接用于真实机器人。

原生 GR00T eval 走 `get_action`，不走训练 Flow-SDE 分支，也不加 physical action noise。
但关闭这两项探索噪声不等于完全确定性：不能仅凭 env seed 相同就声称每次 policy 输出逐位相同。
最终排名需要重复评测并记录 seed；若要求严格配对 policy 初始噪声，须另行验证原生 worker 的 RNG 控制。

实现依据：`rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py` 的
`sample_mean_var_val`、`get_rl_action`、`_predict_normalized_action`、`_apply_exploration_noise`，
以及 `rlinf/models/embodiment/gr00t/simulation_io.py` 的 G1 action converter。

### 评测与继续训练的判断

1. **先做两个 WM 的 SFT base 基线。** 尤其新 scratch WM 目前只有静态检查，还没有实际 GPU 加载和完整 eval。
   使用相同 55 个固定验证案例、56 个计算槽位（padding 不计成功率）、35 步 WM、原生 eval 模式并保存视频。
   最好同一版本下重测旧 WM；历史成功率只作参考，不混用修复前后的结果。
   新 WM 若基线表现异常，先用相同 recorded actions 比较 teacher-forced / closed-loop 与奖励概率，再决定是否开始 RL。
2. **六组各一个训练 seed 做首轮筛选。** 每 5 代正常 eval，20 / 50 代是复盘点而非自动结束点。
   比较同样的累计轨迹数，并记录 GPU 小时；不要只对比某一代 train success 或最小 policy loss。
   训练首代的 rollout 发生在首次 actor update 之前，可用来比较不同噪声的起始探索，但不是训练后 eval。
3. **主要指标是固定验证集的成功率和三阶段完成率。** 同时看 `group_flat_fraction`、
   `group_zero_fraction`、`group_return_std_mean`、KL、clip fraction、grad norm、动作范围/抖动及视频。
   success_once=0 不等于无学习信号：同组回报有 0 / 1 / 2 的阶段差异也可以产生 GRPO advantage。
   反之，同组全都拿到相同非零分数也没有组内区分信号。噪声使组内回报差异增大，也可能只是 WM/奖励不稳定；
   需要结合视频与 eval，不能只追求较大的 std。
4. **不只在各自训练 WM 中排名。** 候选策略在两个 WM 中交叉 eval，并同时报告 SFT base 在这两个 WM 上的分数。
   比较同一评测 WM 下的策略提升，避免把“一个 WM 更容易骗过 reward classifier”当成 RL 更好。
   两个 WM 都不是独立物理真值；最终仍需真实环境检验，并人工核对视频与三阶段奖励的对应关系。
5. **重复与留出测试。** 首轮六组不是充分的多 seed 结论；选出前两组后补到每组三个训练 seeds，例如
   `(actor.seed, env.train.seed)=(1234,0),(1235,1),(1236,2)`，评测协议保持统一。
   预算充足时完整矩阵为 6 条件 × 3 seeds = 18 次训练。55 案例中一例就约 1.82 个百分点，
   小幅领先不能靠单次 eval 定输赢；反复用于选参数的 55 例是 validation，不是最终独立 test set。
6. **提前结束只依据持续信号。** 连续多个 eval 点显著退化、组内回报长期几乎全相同且无验证改善、
   动作/视频明显异常或非有限梯度时先排查，必要时停止该组；不要仅凭一次低 train success 砍掉高噪声组。

当前 chain 恢复 actor/optimizer/LR 等训练状态，但不恢复 env/rollout 的完整 RNG/轨迹状态；
跨 Slurm 续跑不等同逐轨迹确定性复现。记录 chain 边界，不能把独立重启当成额外 seed 重复实验。

## 与 Wan RL 配方的差异

对照本仓库 `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml` 及其 env/model defaults；
object / goal 配方在下表关键超参数上相同。
[官方 Wan 示例](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)
也给出 OpenVLA-OFT、8-action chunk、5 步 WM 与 KIR 等设置。下面以本地代码实际参数为准，不把另一架构的推荐直接移植。

| 项目 | Wan / OpenVLA-OFT 配方 | 当前 DreamDojo / GR00T cluster 方案 | 处理建议 |
| --- | --- | --- | --- |
| 算法骨架 | GRPO、group=8、normalize advantages、无 critic、KL/entropy 系数 0、clip=0.2/0.28 | 相同 | 主矩阵保持 |
| 策略探索 | 离散 action token sampling，temperature_train/eval=1.6 | GR00T Flow-SDE noise_level；physical noise=0 | temperature 与 Flow-SDE 没有数值对应关系 |
| 采样总量 | 64 env × 16 rollout = 1024 条/代，128 组 | 64 × 2 = 128 条/代，16 组 | 我们每代采样量为其 1/8，不能只按代数比较 |
| Actor global / micro batch | 8192 / 32 | 128 / 8 | global 是 chunk 样本的更新批量，不是 WM 并行 env 数 |
| Actor lr | 2e-5 | 5e-6 | 不在六组中同时调 LR |
| Adam beta2 / eps | 0.999 / 1e-5 | 0.95 / 1e-8 | 记录差异，不直接认定现值错误 |
| Reward / 筛选 | reward_coef=5，filter_rewards=true，组平均回报区间 [0.5,4.5] | coef=1，三阶段递增奖励，总回报最多 3，filter=false | 阈值含义不同，不能照抄 |
| WM 推理 | 5 步，256×256，5 条件帧 + 8 未来帧 | 35 步，480×640，1 条件帧 + 12 未来帧，执行其中 6 帧 | 主矩阵保持35；另外做质量/速度实验 |
| 初始状态 | KIR=true，支持关键帧初始化 | KIR=false，random_start_frame=false | DreamDojo 当前强制 KIR off，不能直接打开 |
| 动作/观测 | 7D action，chunk=8，不用 proprio；train horizon=256 | 28D 绝对关节目标，chunk=12，带 proprio 代理；horizon=240 | 不移植动作尺度/chunk/horizon |
| Eval 环境 | env.eval 指向物理 LIBERO，而非 Wan | 当前仍是 DreamDojo WM | 跨 WM eval + 最终真实环境验证 |
| Eval 频率/规模 | val_check_interval=-1（定期 eval 关闭）；eval 配置 496 env | 每 5 代，55 有效案例 | 保持原生每5代，避免额外高频全评测 |

GR00T `predict_action_batch` 当前丢弃通用 sampling kwargs，实际由 noise_level / action_noise_scale 控制上述训练探索。
因此当前 YAML 的 temperature_train=1.0 / temperature_eval=0.6 不等价于在本任务里调探索强度，
也不建议为“对齐 Wan”只把它们改成 1.6。

### 优先于增大 physical noise 的后续实验

**第一优先：actor global batch 128 → 256 / 512，保持 micro=8、env=64×2。**
当前每代 actor 样本数是 `128 trajectories × (240/12) chunks = 2560`；默认 update_epoch=1 时：

| global batch | 每代 optimizer steps | 8 卡下每卡梯度累积次数（micro=8） |
| --- | --- | --- |
| 128 | 20 | 2 |
| 256 | 10 | 4 |
| 512 | 5 | 8 |

Wan 对应 `64×16×(256/8)=32768` 个 chunk 样本，global=8192 时每代更新 4 次。
我们每代收集更少轨迹，却拆成更多次更新；这是值得测试的稳定性差异，不是已经证实的 bug。
256 / 512 都满足当前整除约束，且不增加 WM 同时运行的 env 数；micro 不变，不通过扩大一次前后向去占更多显存。
更大 global 也改变每代 optimizer steps，不能保证收敛一定更好。先固定 LR=5e-6 做独立对比，
再视 eval / KL 考虑单独比较 LR=5e-6 / 1e-5，不与主 noise 矩阵一起改。
依据：`rlinf/workers/actor/embodied_fsdp_actor_worker.py` 的 `run_training`。

**第二优先：WM 的速度/质量，而不是再扩大同时运行的 env 数。**
基准 job 1105305 的 rollout 占 967.25 / 1080.83 秒，约 89.5%，actor 约 64.67 秒。
在选出的 WM/噪声组合上，可以另测 train WM 35 / 15 / 5 步，eval 固定 35 步；
先通过相同 recorded actions 的闭环视频、三阶段奖励和短 policy rollout 检查。
不能因为 Wan 用 5 步，就假定两个 DreamDojo checkpoint 均能用 5 步而不损伤训练信号。
按同样轨迹量和同样 GPU 小时分别报告，区分样本效率与运行效率。

**第三优先：如果卡在交接/放置阶段，研究初始化覆盖与奖励有效性。**
先按数据来源和初始案例统计阶段瓶颈，检查验证数据是否独立于 WM/reward 训练数据。
可在独立实验中研究从真实演示的中间阶段初始化，但必须一起恢复对应 state、action history 与奖励阶段，
避免在已经拿起/交接的图像上免费发前序奖励；这需要单独实现/验证，不是直接把 enable_kir=true。
现有 `filter_rewards` 按**组平均回报**掩蔽 loss，不是按组方差筛选，也不会自动补采样。
例如全部成员回报都为 1 的组仍可能通过区间筛选，却没有组内优势；先用现有组统计诊断，不照搬 Wan 阈值。

## 时间预算与记录

以已经完成的 8×H100、64 env × 2、35 步旧 WM 单代测试约 18 分钟为粗略基准，
20 代约 6 小时/组，六组约 36 node-hours（288 GPU-hours）；50 代约 15 小时/组，六组约 90 node-hours。
这些估算**尚未包括定期 eval、checkpoint、容器初始化、链式重启及排队**；新 rank 32 WM 的实际速度尚未验证。
六组若全跑 1000 代，单组仅训练部分也约 300 小时，需根据曲线择优继续，不建议无监控地全部跑满。

沿用 CHAIN_TIMEOUT=3.9h；本轮 sweep 显式设置 `MAX_RUNS=18`，包含首份作业，最多 18×4=72 节点小时/组，
不能保证完成 1000 代。通用 chain 脚本默认仍为 6，不修改已验证的续跑代码。
每 5 代保存及续跑边界保存会产生较大的原生 checkpoint，需预估存储，
本次不自动删除任何旧产物。
记录实验名、三类 seeds、WM/SFT/reward 校验值、解析后的原生 config、容器/runtime 版本、chain job ids、
累计轨迹数和 GPU 小时即可；不需要在每个 log 中重复保存 source 与 `.diff`。
已有时间/指标原始出处见 [BATCH_SCALING.md](BATCH_SCALING.md)，续跑限制见 [SLURM_CHAIN.md](SLURM_CHAIN.md)。

## 已完成与未完成

当前完成：定位、训练元数据和推理结构核对、SHA256、统一路径接入。
本机 Hydra 静态组合检查通过：两个 WM × noise=0.1 / 0.3 / 0.5 六组均解析成功，
train/eval 各自使用同一权重及匹配的 experiment，WM=35 步、GR00T=4 步、physical noise=0、
actor lr=5e-6、64×2/global128/micro8、1000 代与 val/save=5 均符合矩阵。
另检查了 2560 个 chunk 样本与 global batch=128 / 256 / 512、8 ranks、micro=8 的整除约束，全部通过。
这不是 GPU 容量或收敛测试。旧默认配置没有被覆盖；现有 MSC 可选配置源警告不影响本次静态解析。
本轮补充：38 项 CPU 测试通过（sweep 6 项、原有 chain 32 项），bash 语法、Ruff 检查通过。
主矩阵及备用 global batch=256/512 合计 18 份 Hydra 配置均静态解析通过；六组主实验已提交。
提交后首次检查为 5 个 RUNNING（启动阶段）、1 个等待 Resources，不能据此宣称完整单代训练或 eval 通过。

## 本轮提交与备用 batch 配置

首批按用户要求只启动原六组，没有同时提交 global batch=256/512；后续追加授权与作业见文末。
首份作业于 2026-09-11 提交，单节点 8 GPU、4 小时 allocation、CHAIN_TIMEOUT=3.9h、MAX_RUNS=18、
runner.max_epochs=1000、runner.max_steps=-1，从相同 SFT base 初始化；不是 20 代短测。

| 实验名 | 初始 Job / Chain ID |
| --- | --- |
| wm_r64_n01_gb128_s1234 | 1106598 |
| wm_r64_n03_gb128_s1234 | 1106599 |
| wm_r64_n05_gb128_s1234 | 1106600 |
| wm_scratch_r32_n01_gb128_s1234 | 1106601 |
| wm_scratch_r32_n03_gb128_s1234 | 1106602 |
| wm_scratch_r32_n05_gb128_s1234 | 1106603 |

新增入口：`docker/dreamdojo/submit_wm_comparison.sh`。只拼接同一原生 YAML 的 Hydra overrides 并提交现有 chain，
不新增 trainer/evaluator、不复制六份完整 YAML、不重建 sqsh。默认只打印命令，显式 --submit 才调用 sbatch。
提交记录有目录锁，同名已记录的实验不会重复提交。若 sbatch 响应丢失而没写入 jobid，须先检查队列，不能盲目重提。

```bash
# cluster code/RLinf 目录。原六组已经提交，不需再运行 --submit。
bash docker/dreamdojo/submit_wm_comparison.sh --dry-run

# 已准备的额外 12 组：两种 WM × 三种 noise × global batch 256/512；只预览。
GLOBAL_BATCH_SIZES='256 512' bash docker/dreamdojo/submit_wm_comparison.sh --dry-run

# 或只选一个 WM/noise，追加两个纯 batch 对照，复用正在运行的 global=128 基线。
WM_VARIANTS=r64 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES='256 512' \
  bash docker/dreamdojo/submit_wm_comparison.sh --dry-run
```

准确参数是 `actor.global_batch_size`，不是 `env.train.total_num_envs` 或 `actor.micro_batch_size`。
全部预设均 MAX_RUNS=18、max_epochs=1000、micro=8、64 env×2；global=128/256/512 对应每代 20/10/5 次更新。
脚本与测试已同步 cluster；首批提交时脚本 SHA256：`a8974f04dd79bb1c12befcb743d239ed4b9587bab1cac4528dcc0fec19d9431e`。
未修改 chain.py / train_chain.slurm / run_cluster.slurm 或两个 pinned runtime 文件；校验值与真实续跑验收版本一致。

记录根目录：`/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/`。
提交 jobid 和首次 Slurm 日志在 `sweeps/20260911_wm_noise/`；续跑元数据在 `chains/<初始JobID>/`；
每段原生日志、视频和 checkpoint 在 `<当前JobID>-train/`。下一份 job 不是独立实验，首份 job 结束不代表训练结束。

## Wan 的 KIR 具体如何实现

KIR 不是每步将模糊视频替换为真实图，也不是 WM 在线自动判断关键瞬间。在当前 Wan 路径中：

1. 关键帧来自离线准备的 `.npy` 轨迹文件。`NpyTrajectoryDatasetWrapper` 在 enable_kir=false 时
   排除文件名包含 `_kir` 的文件，true 时纳入它们；默认取文件第一帧作 reference、文件末尾四帧作 target_items。
2. `WanEnv.reset` 先将 reference 重复为五张条件图；若有四个 target_items，替换第 2–5 张并填入对应 action history。
   策略看到最后一张观测，可以从预制片段的末端继续，不必每次从完整任务开头重新尝试。
3. 后续每个 chunk 使用固定 reference、最近四张上下文图及对应动作预测未来八帧。
   最近四张随后更新为生成帧，reference 保留；闭环期间没有不断读取未来真实帧纠正结果。

依据：`rlinf/data/datasets/world_model.py` 的文件过滤、`first_frame` / `last_n_frames`、`__getitem__`，
以及 `rlinf/envs/world_model/world_model_wan_env.py` 的 `reset`、`_infer_next_chunk_frames`。
覆盖哪些任务阶段取决于离线关键帧文件，不是布尔开关自动创造的。

DreamDojo 当前是 1 张条件图、384D joint conditioning，并显式拒绝 enable_kir=true。
如借鉴这个思路，应独立实现从真实中间帧重置，正确恢复 state、动作基准/历史、reward stage；
不能照抄 Wan 五条件帧输入，或在已完成拿起/交接的初始画面上免费发前序奖励。本次六组不更改 KIR。

## 追加 scratch WM 的 LR / global batch 对照

用户随后要求在 scratch_lr3e-4 权重下追加学习率与 global batch 实验。
采用单因素对比，固定 scratch rank 32 WM、noise_level=0.3，以已有 1106602 为基线，不再重复提交基线。
学习率是 `actor.optim.lr`，不改变冻结 WM 的参数或其历史训练 lr=3e-4。

| 实验 | actor.optim.lr | actor.global_batch_size | 初始 Job / Chain ID |
| --- | --- | --- | --- |
| 已有基线 wm_scratch_r32_n03_gb128_s1234 | 5e-6 | 128 | 1106602 |
| wm_scratch_r32_n03_gb128_lr1e5_s1234 | 1e-5 | 128 | 1106686 |
| wm_scratch_r32_n03_gb256_s1234 | 5e-6 | 256 | 1106687 |
| wm_scratch_r32_n03_gb512_s1234 | 5e-6 | 512 | 1106688 |

三组仍是 MAX_RUNS=18、max_epochs=1000、max_steps=-1、micro=8、64 env×2、WM train/eval 35 步、
GR00T 4 步、physical noise=0、val/save=5、相同 seeds，分别从 SFT base 新建 chain。
没有同时改变 LR 和 global batch；如要测试二者交互，须另行提交组合组。

已使用的命令（已提交，不需重复运行）：

```bash
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES=128 ACTOR_LRS=1e-5 \
  bash docker/dreamdojo/submit_wm_comparison.sh --submit
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES='256 512' ACTOR_LRS=5e-6 \
  bash docker/dreamdojo/submit_wm_comparison.sh --submit
```

提交器新增 ACTOR_LRS 选择，默认仍为 5e-6，不改变默认六组；非默认 LR 的实验名额外带 _lr1e5，避免覆盖或跳过基线。
更新后 7 项 sweep CPU 测试、Ruff/bash 检查通过，三组 Hydra 静态解析通过。
更新后提交器 SHA256：`92f7bada99f41dc418975f9108aedf223c4aac3be14907d98c47f507e735076b`。
只更新宿主机提交器，不修改运行中的训练源码或 chain hash 涉及的文件。

追加时原六组均 RUNNING；原六组日志已有 WM 加载结果 missing_keys=[]、unexpected_keys=[]、incorrect_shapes=[]，
LAM 有严格恢复记录，针对 CUDA OOM、shape mismatch、RuntimeError、WorkerCrashedError 的扫描无命中。
这不是完整单代或 eval 验收；新增三组提交后等待调度，状态以实时 squeue 为准。

### 后续状态核验：2026-09-12 04:44 UTC

九组均在自动续跑，当前 squeue 全部 RUNNING。新增三组的首份作业均 COMPLETED 0:0，
且已有完成标记指向后续作业保存的 checkpoint，证明不是仅提交后一直排队。

| 条件 | 初始 Chain ID | 当前运行 Job | 最近已发布完成标记的 checkpoint |
| --- | --- | --- | --- |
| scratch n03，lr=1e-5，global=128 | 1106686 | 1118101 | step 35（保存于 1113228） |
| scratch n03，lr=5e-6，global=256 | 1106687 | 1117864 | step 35（保存于 1113092） |
| scratch n03，lr=5e-6，global=512 | 1106688 | 1118027 | step 36（保存于 1113012） |

checkpoint 步数不是当前实时训练步数；正在运行的段可能已继续前进。
三组 launch.json 实际参数已再次核验：scratch 权重、noise=0.3、各自 LR/batch、max_epochs=1000；
初始日志明确 link 1/18，NCCL_P2P_DISABLE=0、NCCL_P2P_LEVEL=unset。
原六组当前 job 分别为 1117877 / 1117941 / 1117920 / 1117895 / 1117916 / 1117937（按上表初始六组顺序）。
本次核验没有修改训练参数或重复提交任何任务，没有据此对学习效果作结论。
