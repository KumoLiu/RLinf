# 独立 RLinf / DreamDojo / GR00T 容器

2026-09-11：按本机已验证的运行环境新建镜像，不继承、修改或覆盖 cluster 的 DreamDojo 镜像。
部署代码位于 `docker/dreamdojo/`。已完成环境、资产迁移、一轮原生短训练及完整 SFT eval，不启动长训练、不新增 CI，
不修改 GR00T 的 RL 去噪逻辑。新增的可选原生 runner 预算退出及 Slurm 自动续跑见
[SLURM_CHAIN.md](SLURM_CHAIN.md)；原生训练更新和 eval 流程不变。

## 环境基线

| 项目 | 固定版本 / 来源 |
| --- | --- |
| 基础镜像 | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`，Dockerfile 固定 digest |
| Python | 3.10.21，`/opt/venv/bin/python` |
| PyTorch | 2.7.0+cu128 |
| FlashAttention | 2.7.3+cu128.torch27 |
| Transformer Engine | 2.2+cu128.torch27 |
| natten / xformers | 0.21.0+cu128.torch27 / 0.0.30 |
| Transformers / Ray | 4.57.6 / 2.58.0 |
| GR00T 源码 | `2d9a9fce9811d10510d1263344fbf5253f33639d` |
| RLinf / DreamDojo | 本机含已验证补丁的源码快照；完整哈希见镜像内 `source-manifest.json` |

342 个非 editable 包的版本记录在 `requirements.txt`、`cuda-wheels.txt`、
`runtime-versions.json`；CUDA 专用 wheel 还固定下载地址和 SHA256。
源码以 editable 形式安装到 `/opt/src/{RLinf,DreamDojo,Isaac-GR00T}`。
不把环境放在 `/root` 或用户 home 下，避免被 Pyxis 的 home 挂载遮盖。

这是**复现本地验证环境**，不是声称满足新版 GR00T 的全部官方依赖声明：
上游声明 Python 3.12 / Torch 2.9，本机升级只更新了 GR00T 源码，保留 Python 3.10 / Torch 2.7。
安装时明确使用 `--no-deps`，GR00T 另用 `--ignore-requires-python`；
不通过重新解析上游依赖偷偷升级 Torch。依赖元数据冲突仍需记录，不能以 `pip check` 全绿代替实际验证。

两个容易误改的本机细节也保留：torchvision 包版本 0.22.0 的运行时标签为 `+cu126`；
两个 OpenCV distribution 同时存在，最终生效的是 `opencv-python-headless` 4.12.0.88，
因此镜像安装时最后重装它，并断言 `cv2.__version__ == '4.12.0'`。

## 构建、验证和导出

在本机 RLinf 根目录执行；每次使用新的 context 目录和镜像版本，不覆盖旧产物。

```bash
.venv/bin/python docker/dreamdojo/prepare_context.py \
  --output /localhome/local-yunl/container_builds/rlinf-dreamdojo-20260911-v1/context \
  --dreamdojo /localhome/local-yunl/DreamDojo \
  --gr00t /localhome/local-yunl/RLinf/.venv/gr00t

sudo docker build --progress=plain \
  -t rlinf-dreamdojo:gr00t2d9a9fc-cu128-v1 \
  /localhome/local-yunl/container_builds/rlinf-dreamdojo-20260911-v1/context

# 本机物理 GPU 4 故障，只透传健康的 GPU 0。
sudo docker run --rm --device nvidia.com/gpu=0 --ipc=host \
  rlinf-dreamdojo:gr00t2d9a9fc-cu128-v1 \
  python /opt/rlinf-build/smoke.py --gpu

bash docker/dreamdojo/export_sqsh.sh \
  rlinf-dreamdojo:gr00t2d9a9fc-cu128-v1 \
  /localhome/local-yunl/container_builds/rlinf-dreamdojo-20260911-v1/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh

bash docker/dreamdojo/upload_sqsh.sh \
  /localhome/local-yunl/container_builds/rlinf-dreamdojo-20260911-v1/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh
```

`prepare_context.py` 拒绝与记录不一致的包版本、非预期 GR00T commit、源码软链接，
只复制列出的运行时代码目录，不复制 `.git`、`.venv`、日志、模型权重或隐藏凭据。
本次 context 约 18 MB。构建产物和构建日志统一放在 `container_builds/`，不混入实验日志。
context 是生成产物；后续源码改动要重新准备 context，不能误以为已自动进入旧镜像。

`export_sqsh.sh` 使用 Enroot 的 `dockerd://` 导出，限制压缩线程数并产生 SHA256。
`upload_sqsh.sh` 先传 `.partial`，校验成功才改为最终文件名；拒绝覆盖同名不同内容的镜像。
镜像不含训练数据、权重或 HF token。

2026-09-11 的实际产物：Docker manifest `09f50b1695245dfc20f2d0b3cae1d3ff034146873fb583149787453ec90d14df`；
sqsh 约 11.98 GiB，SHA256 为
`a763b639bd5def81c9ee9da484d4483eedbd3fadd474e728b50572234ee3c985`。
本机容器 GPU 0 检查已通过（H200 NVL，FlashAttention/Transformer Engine 的 BF16 前向和反向）；
约 08:12 UTC 开始上传。传输日志为
`/localhome/local-yunl/container_builds/rlinf-dreamdojo-20260911-v1/upload.log`。
上传已完成，远端 SHA256 与上述值一致，最终 `.sqsh` 已就位；传输耗时约 1 小时 46 分钟。
cluster 基础容器/GPU smoke 已通过（job `1100464`）；补充下文说明的 LAM 源码后，
实际模型加载及一轮原生 GRPO 训练也已通过（job `1103749`）。

## cluster 目录与挂载

cluster alias：`nvidia-cluster`；account：`healthcareeng_isaac`。
统一父目录：`/lustre/fsw/portfolios/healthcareeng/users/yunl`。

| cluster 路径（相对父目录） | 容器路径 | 用途 |
| --- | --- | --- |
| `docker/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh` | 镜像本体 | 独立于已有 DreamDojo sqsh |
| `rlinf_assets/20260911/data` | `/assets/data`，只读 | 六个原始 LeRobot 数据集 |
| `checkpoints/` | **保持宿主机相同的绝对路径**，只读 | 统一复用 SFT / WM / reward / LAM，不复制到实验目录 |
| `cache/huggingface/hub` | `/cache/huggingface/hub`，只读 | 复用公开模型文件缓存，不挂载 token |
| `outputs/rlinf` | `/outputs`，可写 | 每个 Slurm job 的结果 |
| `code/RLinf` | 默认不挂载 | 可编辑的部署源码副本；镜像使用其内部固定快照 |

权重统一管理在现有 `checkpoints/`，不打进镜像，也不复制到 `rlinf_assets/`。
`GR00T_MODEL_PATH` 直接指向已有的
`checkpoints/gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127`。
三个 safetensors 分片，以及 config、processor_config、statistics、embodiment_id
均已 SHA256 核对，与本机完全一致。
原有 `processor/` 子目录由 RLinf 现有加载器支持，不需要搬文件或修改加载逻辑。

WM 与 reward 已准备到统一目录：

- `checkpoints/DreamDojo/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt`
- `checkpoints/reward/milestone_v2/best.pt`

WM 直接复用已有
`outputs/train/dreamdojo/hf_teleop_rollout_posttrain_lora/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt`，
校验与本地一致后，在上述 `checkpoints/` 目标建立**硬链接**：同 inode、link count=2，
没有重新上传、复制或移动 4,486,867,523 字节的权重，也没有改写原文件。
硬链接不是独立备份，两条路径不能用于分别修改权重；训练只读加载，输出另存。
WM SHA256：`267f2209e3e58fec5a153d8a92984e05a60005f74661e2e6555e40e1b66147b0`。

reward 是唯一新增上传的权重，44,900,363 字节；先写 `.partial`，校验后改为 `best.pt`。
SHA256：`2dd843e6a78a72a2af626bbdcf7bc9df182ac68f0119d45f1d585e17f93080d7`。
原有 checkpoint 没有被覆盖。训练/eval 启动前检查所有 checkpoint 存在，
且位于只读挂载的 checkpoint root 下；原生训练新产生的 checkpoint 仍写入独立实验输出。

数据集名称见 [README](README.md)。同步使用 `rsync -aL` 解引用本地视频软链接，
meta、parquet 和视频一起迁移；不使用 `--delete`。SFT 需要完整 processor/config/statistics，
不能只复制 safetensors。原始 RL reset 数据不是已有 `g1_hf_*` WM 训练数据。
HF hub 已有 Cosmos Predict2.5 的 2B/14B 和 Reason1/Reason2 缓存。
与本地缓存执行 `rsync --dry-run --size-only` 对比：78 个目录/文件/链接中没有缺文件、大小差异或
不同 snapshot 链接，预计传输内容为 0 字节；未执行写入同步或更改共享缓存权限。
Reason2 `refs/main` 同为 `9ce19a195e423419c349abfc86fd07178b230561`。
这是存在性/大小/链接检查，不等于所有约 59 GB 缓存已逐字节 hash 校验；最终仍以离线加载为准。

### 数据复用核对及 episode 2 修订

首次完整上传六个原始数据集约 264 MiB，存在不必要的重复：225 条 teleop train 的视频、
parquet 已在 `datasets/pick_trocar_teleop_success_train`，逐文件 checksum 一致；只有
`meta/stats.json` 存在统计字段/精度差异，当前 RL reset 加载器不读取此文件。
其余五组 `g1_hf_*` 的 parquet 是 43D，不能直接替代 RLinf 使用的 28D 原始数据；
但总计 546 个视频中，修订前已有 545 个与本机一致。后续应统一使用 `datasets/`、复用视频，
仅补原始 parquet/meta，而不是再复制完整数据集。本次优先启动镜像传输，尚未迁移或删除首次上传副本。

用户确认 30k val 的 episode 2 在本机已裁剪，授权替换 cluster 旧版。现已更新
`datasets/g1_hf_pick_trocar_rollouts_30k_val/videos/chunk-000/observation.images.cam_head/episode_000002.mp4`：

- 496 → 145 帧；视频 SHA256：`bb9a836efb351756d0bf2d69eb1d7ce0dcd89c10ebb765cc45be3b8ec3bc5714`。
- 同步裁剪 episode 2 的 43D parquet，更新总帧数 2432 → 2081 和相关统计。
- episode 3–16 只前移全局 `index` 351 帧及其统计；所有保留的非 index 列均逐值验证不变。
  episode 0–1 的 parquet 不变，其他视频不变；没有修改 WM 的训练归一化文件 `G1_stats.json`。
- 使用已有 `DreamDojo/scripts/milestone/trim_episode.py` 的裁剪函数；额外修正后续 episode 的
  index 统计，以及被裁视频的图像统计；解码验证新版确为 145 帧。
- 共更新 20 个文件，更新前旧 SHA256 / 更新后新 SHA256 均通过校验。
- 旧文件及变更清单保存在 cluster 的
  `dataset_backups/20260911_30k_val_episode000002/`，可恢复；没有删除旧版。
  本地准备脚本、暂存与 `case-update-manifest.json` 位于上述 `container_builds/...-v1/` 下，
  没有把一次性数据修订脚本加入 RLinf 运行时代码。

## Slurm：先验证，后运行原生入口

**v1 镜像的已知打包遗漏及临时补充：**首个真实 WM 初始化发现镜像没有
`external/lam`。已修复 `prepare_context.py` 的源码清单，并将 LAM import 加入 CPU smoke、
深层 action-conditioned WM import 加入 GPU smoke（后者在 import 时检查 CUDA/FlashAttention）。
后续重新构建的镜像会包含修复；当前上传的 v1 sqsh 没有被修改或重新上传。

早期 job 使用本机原样的 5 个源码文件 / 24,924 字节，单独同步到
`code/RLinf-runtime/20260911-lam/external/lam`；`rsync --checksum --dry-run` 校验无差异。
这只解决 import，不解决权重缺失后静默继续。后续用户授权的严格加载器已部署到新目录
`code/RLinf-runtime/20260911-lam-strict/external/lam`（5 文件 / 25,908 字节），旧目录保留。
由显式的 `DREAMDOJO_EXTERNAL_ROOT` 指定，只读挂载到 `/opt/src/DreamDojo/external`；
仅修改 LAM 权重加载，不改 WM/G1 推理。LAM 直接复用 `checkpoints/DreamDojo/LAM_400k.ckpt`。
v1 作业提交前须设置下面的环境变量；未来包含严格加载器的完整镜像可不设置 supplement。
详细证据、hash 和验证状态见 [LAM_CHECKPOINT_AUDIT.md](LAM_CHECKPOINT_AUDIT.md)。

```bash
# 在 cluster 宿主机设置，供 Slurm 构造只读挂载（不是容器内部路径）。
export DREAMDOJO_EXTERNAL_ROOT=/lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf-runtime/20260911-lam-strict/external
export DREAMDOJO_LAM_CHECKPOINT=/lustre/fsw/portfolios/healthcareeng/users/yunl/checkpoints/DreamDojo/LAM_400k.ckpt
export NCCL_P2P_DISABLE=0
unset NCCL_P2P_LEVEL

# 在 cluster 的 code/RLinf 下。默认是 30 分钟、单节点 8 GPU 的环境检查。
sbatch docker/dreamdojo/run_cluster.slurm smoke

# 独立实际 LAM 加载/前向与八卡通信检查，不运行 policy 或训练。
# 需从 code/RLinf 提交；其他目录可显式设置 DREAMDOJO_VERIFY_SCRIPT 绝对路径。
sbatch --time=00:15:00 --job-name=rlinf-dd-lam-p2p docker/dreamdojo/run_cluster.slurm verify

# 环境及资产验证通过后才运行；该入口是 RLinf 已有的独立 eval。
sbatch --job-name=rlinf-dd-eval docker/dreamdojo/run_cluster.slurm eval

# 以下只是后续长训练示例，本次不执行。普通 batch 的上限为 4 小时。
sbatch --time=04:00:00 --job-name=rlinf-dd-train \
  docker/dreamdojo/run_cluster.slurm train
```

脚本默认使用被分配的整节点 8 GPU，placement 为 0–7。本机 GPU 4 故障不适用于 cluster。
train 覆盖为 64 env / global batch 64，group 8；eval 仍为 56 槽位、55 个唯一案例。
这是并行配置变化，与本机 56-env 训练不是严格同 batch 对比。仅申请部分 GPU 时，
必须同步覆盖 placement 和 env/batch，不能访问分配以外的卡。

`train` 仍调用已有 `run_dreamdojo_trocar.sh`，`eval` 仍调用已有
`evaluations/eval_embodied_agent.py` 并设置 `rollout.model=${actor.model}`。
不会创建新的评测进程池，不会自动续交任务。cluster 默认 `NCCL_P2P_DISABLE=0`，
不设置 `NCCL_P2P_LEVEL`；显式 0 防止旧镜像中本机包装器回退到 1。
本机包装器的故障规避不变。历史 job 1103749、1104331、1104860 均在禁用 P2P 下运行，
不能当作正常 P2P 下的性能基线。新增 `verify` 仅用于独立诊断，不替代原生 train/eval。

可通过 `CONTAINER_IMAGE`、`RLINF_ASSETS`（仅数据）、`RLINF_CHECKPOINT_ROOT`、
`GR00T_MODEL_PATH`、`DREAMDOJO_WM_CHECKPOINT`、`DREAMDOJO_REWARD_CHECKPOINT`、`DREAMDOJO_LAM_CHECKPOINT`、
`HF_CACHE_ROOT`、`RLINF_OUTPUTS` 覆盖路径。
容器内环境变量示例见 [cluster.env.example](cluster.env.example)。

## 本次进度与验证边界

- 扩大总采样量已通过：job `1105305`，64 并行 env × 2 轮、global batch 128、micro batch 8，
  完整单代 22 分 20 秒，128 条轨迹，采样 success 50/128，显存采样峰值 67.78 GiB；
  保存原生 checkpoint 和 16 个视频。完整结果及单代验证边界见 [BATCH_SCALING.md](BATCH_SCALING.md)。
  尚未启动正式长训练。
- 后续 LAM/P2P 修复已部署并验证：独立真实 LAM 前向 + 八卡通信 job `1105106`
  通过（1 分 25 秒），原生 64-env / 单 chunk / 单轮集成 job `1105150` 通过
  （5 分 52 秒）；所有 EnvGroup 都实际恢复 LAM，cluster P2P 开关为 0。
  65 项本地 CPU 回归通过。详细配置、hash、日志和零奖励短测的边界见
  [LAM_CHECKPOINT_AUDIT.md](LAM_CHECKPOINT_AUDIT.md)。以下较早 job 的结果保持原始配置记录。
- 已完成：固定依赖安装、342 项版本核对、13 个核心模块导入、5 项打包/挂载单元测试。
- 本机已安装 Docker / Buildx、NVIDIA Container Toolkit 1.20.0、Enroot 4.2.0；
  基础 CUDA 镜像透传 GPU 0 成功。未升级宿主机驱动或更改本机 Python 环境。
- 首次构建在源码安装阶段因未打包的可选 `cosmos-gradio` workspace 元数据失败；
  用 `uv pip install --no-sources --no-deps` 安装已选源码包后通过，未改变运行时依赖。
- 已完成：最终 Docker 构建、最终容器 GPU 运算检查、sqsh 导出及本地 SHA256；
  镜像内 44 项 CPU 回归通过（15.19 秒，3 项 Hydra 路径提示，无测试失败）。
- 已完成：sqsh 上传及远端 SHA256 复核。
- 已同步：`code/RLinf` 中的运行时代码、配置、测试和部署文档，约 9.95 MB / 1135 个文件；
  未同步 `.git`、`.venv`、历史日志、视频或权重，没有覆盖已有 DreamDojo/GR00T 仓库。
  本次容器运行仍使用镜像内固定源码，不会自动改用宿主机新同步的代码。
- 已通过：cluster smoke job `1100464`，`batch` 分区，1 节点 / 8 GPU，最长 30 分钟；
  只执行镜像中 `smoke.py --gpu`，不启动正式训练。日志位于
  `outputs/rlinf/cluster_smoke_20260911/rlinf-dd-smoke-1100464.{out,err}`。
  实际节点 `pool0-00070`，8 张 NVIDIA H100 80GB HBM3；总耗时 62 秒，容器 srun step 52 秒。
  Slurm 主作业及全部 step 均为 `COMPLETED / 0:0`；日志包含 13 个 `IMPORT_OK`、8 个
  `GPU_OK` 和 `SMOKE_OK {"packages": 342, "cuda_tested": true}`。
  stderr 只有 GPU 频率控制提示及 TensorFlow/oneDNN 信息，无异常退出。
  本机是 H200 NVL、cluster 本次分配的是 H100 80GB，不能直接沿用本机整卡显存余量判断。
- 路径调整：用户要求直接复用 `checkpoints/`，已停止向 `rlinf_assets/.../models/gr00t_sft`
  重复上传；中断时留下约 1.47 GB 的不完整副本及两个小配置文件，不被新配置引用，暂未删除。
  六个数据集已传入数据目录；WM 已用硬链接复用、reward 已上传并校验。原有 checkpoint 没有被覆盖或移动。
- 已提交原生短训练 job `1103300`：`batch` 分区，1 节点 / 8 GPU，最长 1 小时。
  通过原有 `run_cluster.slurm train` 调用原生 runner，CLI 覆盖：
  `runner.max_epochs=1 runner.val_check_interval=-1 runner.save_interval=1 actor.micro_batch_size=8`，
  experiment name 为 `cluster_trial_mb8`。cluster 默认 64 train env / global batch 64、group=8，
  WM 35 步、GR00T 4 步、noise=0.1、240 动作长度均不变，保存 train 视频和 checkpoint。
  本轮不跑完整 eval、不自动续训，也未修改正式 YAML 的长期设置。
  Slurm 日志：`outputs/rlinf/cluster_trial_20260911/rlinf-dd-trial-mb8-1103300.{out,err}`；
  runner 日志根：`outputs/rlinf/1103300-train`。该首轮的实际失败原因见下一条。
- 首轮 `1103300` 实际结果：`FAILED / 255:0`，1 分 59 秒；初始化 WM 时因
  `ModuleNotFoundError: No module named 'external'` 退出，未开始 rollout、backward 或保存 checkpoint。
  这是容器漏源码，不能记作模型效果失败或 GPU 故障。原始日志保留。
- 补充依赖后的本机同镜像测试：断网、只读挂载 LAM 源码与新版 smoke.py，GPU 0 上
  15 个 import（含真实 WM 模块）、342 项版本及 CUDA kernel 前后向全部通过；
  6 项打包/挂载回归测试、Ruff 和 shell 语法通过。
- 历史 job 1103749 的 LAM 权重说明：本机 baseline 也没有相对路径 `checkpoints/DreamDojo/LAM_400k.ckpt`，
  `Video2WorldInference.generate()` 中的 LAM 调用是注释代码；当时只补源码，
  没有额外挂载 cluster 的 8 GB LAM 权重或改变 conditioning，保持该推理路径与本地一致。
  对历史缺失提示、旧 WM 训练成功恢复记录及调用路径的详细核查见 [LAM_CHECKPOINT_AUDIT.md](LAM_CHECKPOINT_AUDIT.md)。
- 修复后重试 job `1103749`，节点 `pool0-00088`；与首轮相同参数，额外显式设置上述
  `DREAMDOJO_EXTERNAL_ROOT`，experiment name 为 `cluster_trial_mb8_r2`。
  日志：`outputs/rlinf/cluster_trial_20260911/rlinf-dd-trial-mb8-r2-1103749.{out,err}`，
  runner 输出为 `outputs/rlinf/1103749-train`。最终 `COMPLETED / 0:0`，15 分 38 秒；
  容器主 step 15 分 31 秒，全部 step 正常结束，GPU 分配已释放。详细结果见下一节。
- 已完成：cluster 原生完整 SFT base eval，见 job `1104331` 的结果。
- 未完成：checkpoint 重新加载/续训、多轮稳定性和多节点训练验证。
  单轮训练跑通不等于模型质量提高，也不等于新版上游依赖组合已全面兼容。

## 一轮原生 GRPO 验证结果：1103749

这轮从 SFT base 开始，不加载旧 step 20；8 张 H100 80GB，64 env（8 组 × 每组 8 条），
global batch 64、micro batch 8、WM 35 步去噪、GR00T 4 步、Flow-SDE noise level 0.1。
使用 train reset 数据；每条轨迹为 240 个 30-Hz 动作，分 20 个 WM chunk。
`runner.max_epochs=1`，仅执行一轮 rollout → reward → advantage → actor update → checkpoint，
`val_check_interval=-1` 禁用本次 eval，`save_interval=1` 保存结果。
这些是本轮 CLI 覆盖，正式 YAML 的长期设置没有被改成单轮或关闭 eval。

| 原生指标 | 结果 |
| --- | --- |
| `env/num_trajectories` | 64 |
| `env/picked` / `handed` / `success_once` | 81.25% / 60.94% / 45.3125%（29/64） |
| `env/return` | 1.875 |
| `rollout/group_return_std_mean` | 0.626612 |
| `rollout/group_flat_fraction` / `group_zero_fraction` | 0.125 / 0.0 |
| `train/actor/grad_norm` | 1.648557，有限且非零 |
| `train/actor/total_loss` | 0.000672156 |
| `train/actor/approx_kl` / `clip_fraction` | 0.00191256 / 0.00715457 |
| `time/generate_rollouts` | 509.61 秒（8 分 30 秒） |
| `time/actor_training` | 116.57 秒（1 分 57 秒） |
| `time/sync_weights` | 25.49 秒 |
| `time/step` | 685.96 秒（11 分 26 秒，含本轮保存/收尾；不含前期初始化） |

7/8 组存在 return 差异，本轮 GRPO 不是所有组均零优势的空更新。
所有 worker 日志未出现 OOM、非有限梯度或跳过 optimizer 的告警。
stderr 有两处 `MSC config source not available` traceback：它们来自可选 MSC Hydra 配置源的
空 profile 探测，随后本地 YAML 正常解析并完成训练，不是本轮训练异常退出。
未为消除这些非致命提示而修改依赖或原生 Hydra 逻辑。

这里的 45.3% 是 **actor 更新前、SFT base 在 WM 内的训练采样、由 reward 模型判定的成功率**，
只有 8 组初始 case，不是 64 个独立 case；不是更新后的独立 eval，更不是真机成功率。
只有一个 scalar 点（TensorBoard 原生 step 0，对应保存后的 `global_step_1`），不能判断 loss
下降趋势，也不能用它声称训练提升了成功率。64-env H100 运行与本地 56-env H200
不是严格同 batch、同硬件的性能/效果对照。

### 保存与资源检查

- 原生 checkpoint：
  `outputs/rlinf/1103749-train/cluster_trial_mb8_r2/checkpoints/global_step_1/actor/`。
  `model_state_dict/full_weights.pt` 为 6,910,740,025 字节；
  `dcp_checkpoint/` 的 8 个 rank 分片均存在、每个约 1.68 GB，另有 4,490,451 字节的 `.metadata`。
  按原生 DCP 保存路径包含模型、optimizer、scheduler 状态；已核对保存成功及文件非空，
  尚未重新加载验证。每次此类完整 checkpoint 约 20.3 GB，长训需留意保存频率和存储容量。
- 视频：`outputs/rlinf/1103749-train/video/train/seed_0/0.mp4` 至 `seed_7/0.mp4`，
  共 8 个 2×4 拼图文件，每个包含该 env worker 的 8 条轨迹，不是只生成了 8 条。
  下载 seed 0 和 7 样本并完整逐帧解码通过：H.264、2560×960、15 fps、121 帧、8.07 秒。
  包含初始帧及 120 个生成帧；30 Hz 是动作频率，不应将这些视频强行改为 30 fps。
  本检查确认可播放和时间尺度正确，不替代视频质量/奖励误判的人工评审。
- 同一已分配节点内运行只读 `nvidia-smi` 监测，无额外 GPU 申请。
  约每 5 秒采样，每卡 146 条；最大读数为 67,553 MiB（65.97 GiB）。
  这是覆盖 rollout/actor/save 的**采样整卡峰值**，不是 CUDA allocator 精确峰值，
  也没有覆盖提交后最初的所有初始化阶段。当前 micro=8 单轮无 OOM，多轮仍需观察。
- Slurm 原始输出、所有 worker 日志、TensorBoard/config、显存采样与两段视频已归档本地：
  `logs/cluster_trial_20260911/1103749/`。没有下载大 checkpoint，没有增加 source/`.diff` 快照。

结论：当前 **v1 sqsh + 显式只读 LAM 源码补充**已验证真实数据读取、离线权重加载、
GR00T rollout、WM 推理、reward、GRPO 更新及原生保存。主要耗时仍是 WM rollout；
micro=8 可作为该配置后续训练的候选值。该短训练结束时尚未启动完整 eval 或多轮训练，不能把本轮结果
当作长期学习效果结论。未来独立镜像需重新构建才会内置本次打包修复。

## 原生 SFT base 完整 eval：1104331

2026-09-11 用户确认继续后提交，节点 `pool0-00127`，`batch` 分区、1 节点 / 8 GPU，
上限 1 小时；实际 `COMPLETED / 0:0`，主作业耗时 10 分 44 秒、容器主 step 10 分 35 秒，
全部 step 正常完成，GPU 分配已释放。未启动长训，未修改模型或原生 eval 实现。
提交前远端 sqsh SHA256 再次与本地导出记录一致；镜像仍为上面的 v1 + 显式只读 LAM 补充。
现有 `test_dreamdojo_native_eval.py` 的 9 项 CPU 测试通过（4.28 秒），含 8 卡覆盖 55 个唯一案例；
只有两条已知的可选 MSC Hydra config source 提示。

实际提交（cluster `code/RLinf` 下）：

```bash
export DREAMDOJO_EXTERNAL_ROOT=/lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf-runtime/20260911-lam/external
export DREAMDOJO_GPUS=0-7
export GR00T_MODEL_PATH=/lustre/fsw/portfolios/healthcareeng/users/yunl/checkpoints/gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127
sbatch --partition=batch --job-name=rlinf-dd-eval-sft --time=01:00:00 \
  --output=/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/cluster_eval_20260911/%x-%j.out \
  --error=/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/cluster_eval_20260911/%x-%j.err \
  docker/dreamdojo/run_cluster.slurm eval \
  runner.logger.experiment_name=cluster_eval_sft_base \
  runner.ckpt_path=null runner.resume_dir=null \
  env.eval.total_num_envs=56 env.eval.rollout_epoch=1 \
  env.eval.num_inference_steps=35 env.eval.video_cfg.save_video=true
```

- 入口为现有 `evaluations/eval_embodied_agent.py` / `EmbodiedEvalRunner`；仅创建 rollout/env，
  不创建 actor 训练组。使用 SFT base，不加载本轮 RL step 1 或错误修复前的 step 20。
- 验证数据为 25 teleop + 17 rollout-30k + 13 rollout-10k，固定初始 dataset index，
  group size=1。56 槽位分为 8×7；只统计 55 个唯一 case，末尾 padding 不计分。
- WM 35 步、GR00T 4 步、240 条 30-Hz 命令、原 milestone v2 reward，保存 15-fps 视频。
  policy 使用原生 eval 去噪路径，不使用训练的 Flow-SDE noise=0.1 探索过程；
  初始采样仍可能随机，本次没有添加固定 policy 种子或修改随机数逻辑。
- 本地同 GR00T 版本的原生 SFT eval 为 25/55（45.45%），详见 [GR00T_UPGRADE.md](GR00T_UPGRADE.md)。
  本次为 H100 / 8×7，本地为 H200 / 7×8；硬件、分片和随机数消费不同，
  只能作为迁移后总体效果的参考，不能声称严格逐例配对或把波动归因为训练/版本提升。
- Slurm 日志：`outputs/rlinf/cluster_eval_20260911/rlinf-dd-eval-sft-1104331.{out,err}`；
  原生输出：`outputs/rlinf/1104331-eval`。本次没有新增 source/`.diff` 快照。

### 完成结果与本地基线对照

| 指标 | cluster 本次（8×H100） | 本地同版 SFT eval（7×H200） |
| --- | --- | --- |
| 有效轨迹数 | 55 | 55 |
| picked | 37/55 = 67.27% | 39/55 = 70.91% |
| handed | 25/55 = 45.45% | 26/55 = 47.27% |
| placed / success_once | 25/55 = 45.45% | 25/55 = 45.45% |
| 平均 return | 1.581818 | 1.636364 |
| 原生 eval 阶段耗时 | 449.195 秒（7 分 29 秒） | 573.808 秒（9 分 34 秒） |

cluster 的 picked / handed / placed 概率峰值跨案例平均分别为 0.569257 / 0.467727 / 0.471057。
最终成功数与本地相同，但中间阶段指标并非完全相同；没有逐 case 配对或多次重复运行，
不能声称输出完全一致、统计等价或硬件带来确定的提速。这里的成功仍由 WM 视频上的
reward classifier 判定，不是真机成功率，也不代表短训练 step 1 的效果。

保存检查：

- 全部 8 个视频已下载并逐帧完整解码通过：H.264、2560×960、15 fps、121 帧、8.07 秒。
  每个 `video/eval/seed_0/0.mp4` 至 `seed_7/0.mp4` 是 2×4 拼图，含 7 个计算槽位和
  1 个补齐布局的黑格；最后一个 worker 的 7 个槽位中另有 1 个不计分的 dataset padding。
  因此黑格、padding 与有效失败 case 不能混为一谈。
- 所有 worker 日志中没有 CUDA error、OOM、ModuleNotFoundError 或非有限值异常。
  只有 EnvGroup / RolloutGroup，没有 ActorGroup，没有梯度更新或新 checkpoint，符合纯 eval 预期。
- 归档位置：本地 `logs/cluster_eval_20260911/1104331/`，包含原始 Slurm 输出、worker 日志、
  TensorBoard/config、metrics 和所有视频；不包含 source/`.diff` 或大权重副本。
  `seed_0_final.jpg`、`seed_7_final.jpg` 是从约 7.9 秒提取的诊断预览。
  本轮已验证可播放与时间尺度，尚未完成画面质量的人工复核或清晰度定量比较。

结论：当前部署已通过单节点原生短训练和完整 SFT eval，未见迁移后最终成功率的大幅下降。
可以进入多轮训练；这不替代 checkpoint 恢复测试或长期稳定性验证。本轮没有启动长训，
也未评测更新后的 step 1。后续长训建议沿用已通过短训的 micro batch 8、64 train env，
保留原生每 5 轮 eval/checkpoint，并观察显存、组内回报差异及独立 eval 趋势。

参考：仓库 `add-install-docker-ci-e2e` 容器指南用于组织配方；按本次任务范围未新增 CI。
导出及 GPU 透传采用 [Enroot import](https://github.com/NVIDIA/enroot/blob/main/doc/cmd/import.md)
和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
提供的接口。
