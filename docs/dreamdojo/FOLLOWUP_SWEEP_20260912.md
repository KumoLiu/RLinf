# 第二轮实验：重复 seeds、邻近参数与 15 步 WM

日期：2026-09-12。沿用原生 train/eval 和现有自动续跑，不做交叉评测。

## 选择依据

本轮提交前，四组保留实验均已有第 49 代 train 指标，最新共同 eval 为第 45 代。
以下是同一组 55 个固定验证案例的 WM 内成功判定，不是实机成功率。

| 原实验 chain | 配置 | eval 35 / 40 / 45 | 最近三次均值 | 历史最高 |
| --- | --- | --- | --- | --- |
| 1106602 | scratch / noise=0.3 / batch=128 | 29 / 30 / 36（各 /55） | 57.58% | 65.45%，第 45 代 |
| 1106687 | scratch / noise=0.3 / batch=256 | 29 / 32 / 23（各 /55） | 50.91% | 58.18%，第 40 代 |
| 1106598 | 原 WM / noise=0.1 / batch=128 | 26 / 29 / 26（各 /55） | 49.09% | 52.73%，第 30、40 代 |
| 1106601 | scratch / noise=0.1 / batch=128 | 26 / 27 / 25（各 /55） | 47.27% | 49.09%，第 40 代 |

因此选 scratch/noise=0.3 的 batch=128 和 batch=256 做 15 步训练对照。
batch=256 在第 45 代明显回落，第二名只是最近三次均值排序，优势不确定；
每组目前只有一个训练 seed，不能据此声称统计显著。

指标来自各 chain 的原生 TensorBoard `eval/success_once`，合并续跑段并将 TB step 加 1
转换为已完成训练代数；`eval/num_trajectories` 均核验为 55。最新数据保存在本地
`logs/cluster_sweep_review_20260912/raw/`，原先 review 的汇总图仍是第 35 代决策快照。

## 六组新实验

全部从同一个 GR00T SFT base 重新开始，`runner.resume_dir=null`、`runner.ckpt_path=null`。
不是从某个最佳 RL checkpoint 接着训练，以免混入之前 35 步训练历史的影响。

| 用途 | noise | actor LR | global batch | actor seed / train-env seed | train / eval WM 步数 |
| --- | --- | --- | --- | --- | --- |
| 基准补 seed 1 | 0.3 | 5e-6 | 128 | 1235 / 1 | 35 / 35 |
| 基准补 seed 2 | 0.3 | 5e-6 | 128 | 1236 / 2 | 35 / 35 |
| 单独降低 noise | 0.2 | 5e-6 | 128 | 1234 / 0 | 35 / 35 |
| 单独降低 LR | 0.3 | 2.5e-6 | 128 | 1234 / 0 | 35 / 35 |
| batch=128 的 WM 加速对照 | 0.3 | 5e-6 | 128 | 1234 / 0 | 15 / 35 |
| batch=256 的 WM 加速对照 | 0.3 | 5e-6 | 256 | 1234 / 0 | 15 / 35 |

固定设置：scratch_lr3e-4 的 iter18000 EMA、LoRA rank/alpha=32、physical action noise=0、
GR00T 去噪 4 步、64 个 train env × 2 个 rollout epoch、micro batch=8、group=8。
eval 使用原 55 个案例和 `env.eval.seed=0`，每 5 代 eval/保存；train/eval 视频均开启。
每组 1 节点、8 GPU，cluster 启用正常 NCCL P2P。

最长训练 1000 代；每段 Slurm 上限 4 小时，`CHAIN_TIMEOUT=3.9h` 提前准备保存续跑，
`MAX_RUNS=18`。18 段是另一项预算上限，不保证能跑完 1000 代。

### seed 的具体含义和边界

提交助手新增 `SEED_IDS`：编号 k 映射到 `actor.seed=1234+k`、`env.train.seed=k`。
原有 seed 0 的命名和设置不变。actor seed 用于训练数据打乱等已有随机过程；
train-env seed 影响 reset 案例抽样与 WM 生成种子，eval-env seed 保持不变。

没有修改 rollout worker 的原生随机数实现，也没有为新组单独引入一套 RNG 机制。
这些是独立训练重复实验，不承诺全流程位级复现；原生 rollout 的随机动作采样、
非确定性 CUDA 运算以及续跑时不恢复 env/rollout RNG，均可能带来额外波动。
尤其 `env.eval.seed=0` 只固定环境一侧，不能当作 policy 推理所有随机数都固定。

## 提交与复现

在 cluster 仓库根目录执行。四条命令分别产生 2、1、1、2 个任务，不把 noise 和 LR 混改。
把 `--submit` 改为 `--dry-run` 可只打印配置、不分配 GPU。

```bash
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES=128 ACTOR_LRS=5e-6 SEED_IDS='1 2' TRAIN_WM_STEPS=35 SWEEP_TAG=20260912_followup bash docker/dreamdojo/submit_wm_comparison.sh --submit
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.2 GLOBAL_BATCH_SIZES=128 ACTOR_LRS=5e-6 SEED_IDS=0 TRAIN_WM_STEPS=35 SWEEP_TAG=20260912_followup bash docker/dreamdojo/submit_wm_comparison.sh --submit
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES=128 ACTOR_LRS=2.5e-6 SEED_IDS=0 TRAIN_WM_STEPS=35 SWEEP_TAG=20260912_followup bash docker/dreamdojo/submit_wm_comparison.sh --submit
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES='128 256' ACTOR_LRS=5e-6 SEED_IDS=0 TRAIN_WM_STEPS=15 SWEEP_TAG=20260912_followup bash docker/dreamdojo/submit_wm_comparison.sh --submit
```

提交台账：`/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/sweeps/20260912_followup/`。
同一台账内用 `.jobid` 防重复提交，并用文件锁防止并发提交。
不要为了重试而换 `SWEEP_TAG`，应先检查任务和台账，避免重复占用 GPU。

本轮仅扩展 `docker/dreamdojo/submit_wm_comparison.sh`，新增单元测试及本文件；
未修改镜像、模型代码、原生 YAML、runner、chain.py 或已固定的续跑代码。
默认提交矩阵仍为原六组；默认 seed=1234、WM=35 的实验命名保持兼容。

验证：17 项提交助手测试通过，连同自动续跑共 49 项测试通过；Ruff 与 Bash 语法检查通过；
六组实际 Hydra compose 均通过。
本地 Hydra 可选 MSC 插件因未配置 profile 打印警告，但不影响这六组配置组合成功。

## 实际提交台账

以下六组均已由 `sbatch` 接受；初始 Job ID 同时是其自动续跑 chain ID。

| Job / chain ID | 原生实验名 |
| --- | --- |
| 1120513 | `wm_scratch_r32_n03_gb128_s1235` |
| 1120514 | `wm_scratch_r32_n03_gb128_s1236` |
| 1120515 | `wm_scratch_r32_n02_gb128_s1234` |
| 1120516 | `wm_scratch_r32_n03_gb128_lr2p5e6_s1234` |
| 1120517 | `wm_scratch_r32_n03_gb128_wm15_s1234` |
| 1120518 | `wm_scratch_r32_n03_gb256_wm15_s1234` |

提交助手 SHA-256：`88f3501a7b8b44e3d46cf7d0fa7af9aa78ce8017ffd65a14e8bec5f0f525095a`。
cluster 同步前确认助手与此前已审阅版本一致，同步后再次核验新 hash；
`chain.py`、`train_chain.slurm`、`run_cluster.slurm` hash 保持不变。

后续查看宿主机上的 chain 目录：
`/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/chains/<chain ID>/`。
实际容器内前缀为 `/outputs/chains/`（`/outputs` 映射宿主机 `outputs/rlinf`）；
每段 train 输出位于 `/outputs/<当前 job ID>-train/`，续跑后不要只看最初的 Job ID。

启动核验：六组 Slurm 均进入 RUNNING，初始节点依次为
`pool0-00049`、`pool0-00206`、`pool0-00065`、`pool0-00311`、`pool0-00132`、`pool0-00170`。
六份 `launch.json` 已发布，actor/train-env/eval-env seeds、LR、batch、train/eval WM 步数、
SFT 与 scratch WM 路径、1000 代上限及不 resume 均核验一致；GPU placement=0–7、
`NCCL_P2P_DISABLE=0`。此时仍处于模型初始化阶段，尚无新组 eval 成绩。

随后已下载六组实际生成的 `tensorboard/all/config.yaml`，逐项断言核验上述参数、
GR00T=4 步和 train/eval 视频开启均通过。每组日志均已有 WM 的空 missing/unexpected/shape
不兼容列表，以及 `Restored LAM ... with 0 missing and 0 unexpected keys`，不是跳过 LAM。
启动时 traceback 均核实来自未配置的可选 MSC Hydra profile，之后继续创建 worker、加载模型；
不将此插件警告误记为训练崩溃。尚未等待第一个完整训练代或 eval 完成。

## 后续判读

- seeds：与原 chain 1106602 在相同训练代数比较三次独立训练，不只挑最好的一次。
- noise / LR：分别与原 chain 1106602 比较，先观察至少多个 eval 点，避免单点早停。
- WM=15：同时比较固定 35 步 eval、rollout 耗时、组内 reward 差异和 train 视频；
  不能因为 15 步视频 reward 更高，就直接当作 policy 更好。速度按实测，不能假设整代加速 35/15 倍。
- 最佳 RL checkpoint 与 SFT base 的跨 WM 交叉评测本轮不提交，按用户要求后续再讨论。
- 原四组继续训练，之前已停止的五组不重启，也不删除其日志或 checkpoint。
