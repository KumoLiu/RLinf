# 运行与复现

本轮整理不改变训练行为、不提交实验。以下命令是后续复现入口；
默认配方与已跑实验的显式覆盖值需分开，结论见 [RESULTS.md](RESULTS.md)。

## 1. 环境与资产

本地已验证解释器：`/localhome/local-yunl/RLinf/.venv/bin/python`。
相邻源码：`DreamDojo/`、`Isaac-GR00T/`。
Cluster 使用独立 RLinf SQSH，不与已有 DreamDojo 环境融合：

```text
/lustre/fsw/portfolios/healthcareeng/users/yunl/docker/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh
```

运行栈固定 Python3.10.21 / PyTorch2.7.0+cu128 / GR00T source2d9a9fc。
完整包版本见 [runtime-versions.json](../../docker/dreamdojo/runtime-versions.json)。
这不是未经修改的 upstream GR00T 新版默认 Python/CUDA 栈；不要随意 uv sync 或自动升级依赖。

Cluster根：`/lustre/fsw/portfolios/healthcareeng/users/yunl`。

| 资产 | 根目录以下路径 / 容器映射 |
| --- | --- |
| 权重 | `checkpoints/`；host/container使用同一绝对路径，只读挂载 |
| SFT | `checkpoints/gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127` |
| scratch WM | `checkpoints/DreamDojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt` |
| r64 WM | `checkpoints/DreamDojo/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt` |
| Reward v2 | `checkpoints/reward/milestone_v2/best.pt` |
| LAM | `checkpoints/DreamDojo/LAM_400k.ckpt` |
| RL原始28D数据 | `rlinf_assets/20260911/data` → `/assets/data` |
| HF模型cache | `cache/huggingface/hub` → `/cache/huggingface/hub`，只读 |
| 运行输出 | `outputs/rlinf` → `/outputs` |

路径变量集中在 [cluster.env.example](cluster.env.example)，不要复制checkpoint到每个实验目录。
该文件是cluster/container路径；不要在本机直接source后误用这些路径运行本地训练。

v1镜像需已有的版本化补充源码：

- `DREAMDOJO_EXTERNAL_ROOT=.../code/RLinf-runtime/20260911-lam-strict/external`：严格LAM加载。
- `RLINF_CHAIN_SOURCE_ROOT=.../code/RLinf-runtime/20260911-chain-v1`：原生runner预算续跑。
- KIR / 独立视频审计分别使用 `DREAMDOJO_KIR_SOURCE_ROOT` /
  `DREAMDOJO_VIDEO_AUDIT_SOURCE_ROOT` 的明确只读快照，二者不能叠加。
  路径和hash以原作业配置/chain manifest为准，不随意覆盖旧目录。

未来重新build时包含当前源码即可；不要为清理文档而重建镜像或修改仍用于复现的旧快照。

## 2. 本地原生入口

在RLinf根目录，先检查Hydra最终配置，不启动模型：

```bash
RLINF_PYTHON=/localhome/local-yunl/RLinf/.venv/bin/python \
bash examples/embodiment/run_dreamdojo_trocar.sh --cfg job --resolve
```

去掉 `--cfg job --resolve` 才启动训练。本地wrapper使用已有site env或相邻目录资产；
本机历史GPU故障的排除和NCCL workaround不是cluster通用设置。

Cluster分配8卡，使用0–7；`NCCL_P2P_DISABLE=0`，不设置 `NCCL_P2P_LEVEL`。
不要把本机排除物理GPU4、禁用P2P的限制带到健康cluster节点。

## 3. 最优已测试实验的明确覆盖值

在cluster RLinf checkout运行。以下只打印提交命令，不创建作业：

```bash
WM_VARIANTS=scratch_r32 NOISE_LEVELS=0.3 GLOBAL_BATCH_SIZES=128 \
ACTOR_LRS=5e-6 SEED_IDS=0 TRAIN_WM_STEPS=15 KIR_ENABLED=false \
SWEEP_TAG=closeout_reproduction \
bash docker/dreamdojo/submit_wm_comparison.sh --dry-run
```

检查输出后，如确实要新开实验，使用**新的唯一SWEEP_TAG**并把 `--dry-run` 改为 `--submit`。
脚本会检查资产、同名submission receipt，固定max_epochs1000、MAX_RUNS18、3.9h续跑预算。
`SEED_IDS=0` 对应actor1234/train-env0；1/2对应actor1235/1236，不要填1234。
默认不指定WM/noise时会生成2×3六组；默认train WM是35，不是上述15。

| 对比轴 | 环境变量 |
| --- | --- |
| WM | `WM_VARIANTS="r64 scratch_r32"` |
| Policy采样噪声 | `NOISE_LEVELS="0.1 0.2 0.3 0.5"` |
| Actor更新批量 | `GLOBAL_BATCH_SIZES="128 256 512"` |
| Actor LR | `ACTOR_LRS="2.5e-6 5e-6 1e-5"` |
| 独立训练重复 | `SEED_IDS="0 1 2"` |
| WM训练去噪 | `TRAIN_WM_STEPS=15` 或35 |
| KIR受控消融 | `KIR_ENABLED=true KIR_PROBABILITY=0.5 KIR_MAX_OFFSET_FRAMES=15` |

一次只改变待测试因素；正式KIR提交需对应runtime snapshot。
WM checkpoint必须配同rank的 `env.train.experiment_name` 和 `env.eval.experiment_name`。
scratch使用 `dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora`，
r64使用同名前缀加 `_lr3e-4_r64`；sweep脚本会成对选择。

## 4. 原生独立 eval

入口始终是 `evaluations/eval_embodied_agent.py`。
`docker/dreamdojo/run_cluster.slurm eval` 只是container/mount包装，
设置 `runner.only_eval=true`、`rollout.model=${actor.model}`，不是另一套评测算法。

在source cluster.env.example并明确WM/experiment之后，用原生Hydra overrides指定：

```text
runner.ckpt_path=<cluster full_weights.pt 的完整路径>
env.eval.total_num_envs=56
env.eval.eval_unique_episodes=true
env.eval.enable_kir=false
env.eval.num_inference_steps=35
env.eval.experiment_name=<与WM权重匹配的experiment>
```

SFT对照将 `runner.ckpt_path=null`，其余协议保持不变。
RL policy权重位于输出目录时已通过 `/outputs` 挂载，传入容器路径：
例如host的 `.../outputs/rlinf/1146382-train/...` 对应 `/outputs/1146382-train/...`。
不要把未挂载的host输出绝对路径传给容器。

保存视频和概率审计使用env配置中的video选项及相应已版本化source；
先用 `--cfg job --resolve` 检查实际key。
评测看 `eval/num_trajectories=55`，不是56槽位；训练/独立eval随机轨迹不保证相同。

## 5. Slurm自动续跑与停止

- `train_chain.slurm` 包装原生训练；`run_cluster.slurm` 本身不会续提。
- `CHAIN_TIMEOUT=3.9h` 是预算，runner会给下一代、eval、保存预留时间，
  在完成代边界保存，而不是到点强杀正在写入的checkpoint。
- 完整恢复使用 `runner.resume_dir=<global_step_N目录>`。
  `runner.ckpt_path=<full_weights.pt>` 只初始化policy，不恢复optimizer/scheduler。
- 完成标记、`training_state.json`、DCP metadata与全部rank分片通过检查后才发布latest。
- 同一chain固定语义配置/source hash；新消融用新chain，不要原地修改旧chain。
- 停止先在已核对的chain目录放 `STOP`，等待边界保存退出；
  紧急取消只针对明确job ID，并检查后继排队job，避免孤儿续跑。
- MAX_RUNS达到上限可能以exit1记录，但明确保留checkpoint；不要当CUDA崩溃。
  普通CUDA/NCCL失败、取消和无进度退出不盲目自动重试。
- 恢复不保证env/rollout随机流逐bit相同，见IMPLEMENTATION。

## 6. 保留的诊断与CPU回归

在RLinf根目录：

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  --noconftest -p no:cacheprovider -c /dev/null -q \
  tests/unit_tests/test_dreamdojo*.py

.venv/bin/python toolkits/world_model/dreamdojo_validation.py --help
```

外部DreamDojo源码默认取相邻checkout，也可显式设置 `DREAMDOJO_ROOT`。
以上CPU测试覆盖bridge、LAM、native eval、缓存/快速路径、续跑、KIR与视频审计，
不需要加载GPU模型。

定位新回归的顺序：

1. parity（原始动作/状态输入的一致性）。
2. `world_model` 的真实动作 teacher-forced / closed-loop，比较 `--steps 5 35` 或15/35。
3. `policy` 加入SFT/RL后比较动作趋势；旧step20是修复前产物，不作对照。
4. `real_reward` 与生成视频概率审计；最后才看整体success。

诊断每次使用新的 `--output`，脚本拒绝覆盖已有目录。
5/15/35是WM去噪迭代数，不是视频长度；`--chunks` 决定动作块数量。
诊断结果不是正式55例native eval成绩。

## 7. 重建环境与查看历史结果

镜像工具集中在 `docker/dreamdojo/`：
`prepare_context.py` 用已验证venv导出依赖与白名单源码（拒绝漂移/旧输出目录），
Dockerfile构建独立镜像，`export_sqsh.sh IMAGE /absolute/new/image.sqsh`
导出并保存SHA256，`upload_sqsh.sh`负责传输。
新镜像先 `run_cluster.slurm smoke`，再 `verify` 检查严格LAM和8卡NCCL；
通过后才开始小规模native eval/train。不要打包token、数据或checkpoint进镜像。

TensorBoard可在本机运行：

```bash
.venv/bin/tensorboard \
  --logdir logs/final_review_20260915/tensorboard/comparison \
  --host 127.0.0.1 --port 6007
```

远端使用SSH端口转发，不必暴露公网。优先看 `eval/success_once`、
`eval/num_trajectories`、train组内回报差异及 `time/step`。
loss下降本身不能证明策略变好；历史网页/曲线目录与checkpoint索引见RESULTS。
