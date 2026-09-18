# 在 Neural Simulator 中用 RL 改进 GR00T N1.7

**主要结果：真机 pick–handover–place 成功率由 SFT 的44%提高到 RL 的67%，提升23个百分点。**
采用 `wm_scratch_r32_n03_gb128_wm15_s1234` 第220代；DreamDojo和分类器冻结，只更新GR00T。
这是用户报告的初步真机结果，尚无完整试验次数及逐次判定表，不声称统计显著性。

目录：[数据概述](#data) · [实验 Pipeline](#pipeline) · [实验过程与结论](#experiments) · [实验结果](#results) · [Cluster 与复现](#cluster)

<a id="data"></a>
## 1. 数据概述

任务是单相机下的拿起、双手交接、放置。先采集teleop，再用两个不同SFT checkpoint
采集真机rollout以增加状态和动作多样性；rollout包含成功及约65%失败轨迹（此前汇报口径）。

| 数据来源 | 训练轨迹 | 验证轨迹 | 用途 |
| --- | ---: | ---: | --- |
| Teleop成功演示 | 225 | 25 | SFT、WM适配、阶段分类器 |
| SFT 30k checkpoint的真机rollout | 151 | 17 | WM适配、阶段分类器 |
| SFT 10k checkpoint的真机rollout | 115 | 13 | WM适配、阶段分类器 |
| 合计 | **491** | **55** | 验证集不进入WM训练 |

汇报中的“约200条teleop + 约200条rollout”是概述；表中是最终WM训练清单，
共72,389帧。分类器因排除一条无有效阶段标注的teleop，实际训练490条。
55条验证数据已用于多次模型选择，不能称为未见过的盲测集。

本地原始数据在 `/localhome/local-yunl/data/`，HF来源为 `nvidia/s2r_data_dev`。
原始28D动作供GR00T/RLinf使用；WM训练时适配到43D G1布局，再构成384D conditioning。
不要把43D适配数据直接作为RLinf原始数据输入。

<a id="pipeline"></a>
## 2. 实验 Pipeline

```text
真机Teleop → GR00T N1.7 SFT → 两个SFT checkpoint → 真机Rollout
     └───────────────── Teleop + Rollout ─────────────────┘
                              ├→ DreamDojo微调 → Neural Simulator（冻结）
                              └→ 手工标注三个阶段 → Reward分类器（冻结）

SFT初始化的GR00T → 动作 → Neural Simulator → 新观测 + 分类器奖励
       ↑                      GRPO更新Policy ←───────────┘
       └→ RL checkpoint 与 SFT base 在大致相同真机条件下对比
```

RL更新GR00T的可训练参数，不更新世界模型或分类器。SFT权重同时保留作独立对照；
本轮 `kl_beta=0`，没有固定SFT KL reference或SFT co-training。
当前模拟器是DreamDojo / Cosmos Predict2.5；没有迁移到Cosmos Simulator / Cosmos3。

<a id="experiments"></a>
## 3. 实验过程与结论

### 3.1 先确认管线可用

完成动作/状态桥与DreamDojo dataloader的数值一致性检查（parity），
再用真实录制动作比较teacher forcing（每块用真实图像重置）与closed loop（接自己的预测），
最后加入SFT policy做原生eval。结果支持WM能产生任务进展，但不证明长期接触物理完全正确。

关键修复包括绝对手臂状态重复累加、物理动作裁剪、conditioning时间对齐、
LAM权重严格加载、组内共享图像被原地修改，以及55个唯一eval case的去重计分。
旧step20来自修复前，已弃用。多env批量WM、批量reward和缓存用于降低耗时。

### 3.2 世界模型与奖励

DreamDojo从官方G1 post-train权重做LoRA适配，比较过LR、rank、batch、训练长度、
数据比例、片段长度及motion-consistency。`scratch`指重新任务适配，不是随机初始化。

| 比较项 | 结论 |
| --- | --- |
| WM LR / rank | LR2–3e-4、rank32较合适；保留LR3e-4/rank32方案。rank64像素MSE略好，未证明物理质量更可靠。 |
| 训练长度 / 数据混合 | 9k延长到18k有收益；提高teleop占比没有一致优势，保留约0.34/0.33/0.33混合。 |
| 片段 / motion-consistency | 25帧方案更慢且未优于13帧；关闭或修改MC没有显示稳定优势。原MC的channel/time轴问题已做单独对比。 |

保留WM主配方：13帧片段、LoRA rank32、LR3e-4、global batch32、18k迭代、EMA推理。
像素误差容易受背景主导，不能直接代表器械或接触精度。

正式RL使用v2分类器，识别pick / handover / place，阈值均为0.8。
pick/hand需15 tick中13次通过，place需2/2；每个新阶段奖励+1，总分最多3。
历史奖励锁存、不因后续掉落撤销；15fps生成帧重复成30Hz输入，place两票可能来自同一帧。
因此高奖励不能替代人工判断。v3曾做AI稀疏标注及候选训练，未证明解决误奖、未部署到RL；
现仅保留正式v2代码，过程简述见[DreamDojo分类器说明](../../../DreamDojo/docs/TROCAR_PROJECT.md)。

### 3.3 RL对比实验

| 对比项 | 当前结论 |
| --- | --- |
| Policy noise | **0.3较好**。同30/35/40代窗口，scratch WM35的noise0.1/0.2/0.3/0.5平均success为47.88/51.52/56.36/40.00%。这是Flow-SDE采样噪声，不是直接给视频加噪。 |
| Actor LR | 1e-5早期较差；2.5e-6有时缓解回落，未超过5e-6主线。 |
| Global batch | 128产生最佳单模型；256在WM35有可用候选，WM15下未稳定改善；512无明显优势。 |
| WM去噪步数 | train15 / eval35主线兼顾效果和速度；含周期eval约10.95分钟/代，WM35对照约18.1分钟。 |
| KIR | handover前初始化，p0.5 / offset15在seed1234有潜力，但其他seed未稳定复现；主线关闭。 |
| Seed / WM选择 | 最佳单模型优势不等于配方稳定优势。两种WM各自在自己的环境中评分，也不能直接据此判断哪种WM更真实。 |

KIR按阶段标签选取已pick、未handover且历史充足的起点，不是统一从前15帧开始。
同组8个env共享起点；同时重建图像、状态和奖励历史，不奖励起点已有进展；eval始终关闭KIR。

主线训练设置：

| 项目 | 设置 |
| --- | --- |
| 采样 | 64 env × 2 rollout epoch = 128条轨迹/代；GRPO group8 |
| 轨迹 | 8秒，240条30Hz命令；12条/动作块，共20块；WM视频15fps |
| Policy / Actor | policy去噪4步；LR5e-6；noise0.3；physical action noise0 |
| 更新 | global batch128，micro batch8；每代2,560个动作块，20次actor更新 |
| WM / Reward | scratch rank32 / iter18000 EMA；train15 / eval35；冻结v2 |
| 周期 | 每5代eval/保存，max_epochs1000；eval为55唯一case，56槽中的padding不计分 |

micro batch控制每卡一次前反向的大小，不决定WM推理env数。
gb256/512每代分别更新10/5次，因此batch对比也改变了更新次数。
普通代约9.4分钟，带eval/保存约17.1分钟，主要耗时是WM rollout（约8.3分钟）。

<a id="results"></a>
## 4. 实验结果

| 评测方式 | SFT base | RL第220代 |
| --- | ---: | ---: |
| 真机，大致相同条件（用户报告） | 44% | **67%，+23个百分点** |
| 独立WM视频eval，v2分类器评分 | 25/55 = 45.45% | 41/55 = 74.55% |

220代来自 `wm_scratch_r32_n03_gb128_wm15_s1234`，**没有KIR**。
这是neural simulator中RL能够带来真机收益的初步证据，不是统计显著性或跨任务泛化证明。
WM在接触/遮挡时可能变形，分类器可能误报；模拟器高分不等于真机成功。

保留五组训练曲线和候选；下表峰值来自训练期eval，不是上面的独立视频eval。
训练预算不同且有择优偏差，仅用于候选保留，不作为公平排名。TB step + 1 = 完成代数。

| 实验（均scratch WM、noise0.3） | 候选代 / job | 原生eval峰值 |
| --- | --- | ---: |
| `wm_scratch_r32_n03_gb128_wm15_s1234` | 320 / 1177445；真机用220 / 1146382 | 46/55 = 83.64% |
| `wm_scratch_r32_n03_gb128_s1235` | 190 / 1178228 | 41/55 = 74.55% |
| `wm_scratch_r32_n03_gb128_kirhandp50_off15_s1234` | 115 / 1151683 | 40/55 = 72.73% |
| `wm_scratch_r32_n03_gb256_s1234` | 175 / 1143729 | 40/55 = 72.73% |
| `wm_scratch_r32_n03_gb128_wm15_kirhandp50_off15_s1234` | 145 / 1173304 | 39/55 = 70.91% |

320代没有独立视频或真机复评，不能取代220代的实测结论。
[视频页面](../../logs/best_ckpt_eval_20260913/compare20260914/update220/view.html)保留220 / SFT / 原始录像，端口6009；
[五组TensorBoard](../../logs/final_review_20260915/tensorboard/comparison/00_best_scratch/)在端口6007。
重点看 `eval/success_once`、`eval/num_trajectories`、组内回报差异和 `time/step`；loss下降不等于策略改善。

<a id="cluster"></a>
## 5. Cluster 与复现

### 环境和资产

SSH：`nvidia-cluster`。根目录：`/lustre/fsw/portfolios/healthcareeng/users/yunl`。
使用独立RLinf SQSH，Python3.10.21 / PyTorch2.7.0+cu128 / GR00T源码2d9a9fc；
不要随意 `uv sync`。完整版本见 [runtime-versions.json](../../docker/dreamdojo/runtime-versions.json)。

| 资产 | 相对cluster根的路径 |
| --- | --- |
| 镜像 | `docker/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh` |
| SFT base | `checkpoints/gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127` |
| scratch WM | `checkpoints/DreamDojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt` |
| r64 WM | `checkpoints/DreamDojo/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt` |
| Reward / LAM | `checkpoints/reward/milestone_v2/best.pt` / `checkpoints/DreamDojo/LAM_400k.ckpt` |
| 数据 / 输出 | `rlinf_assets/20260911/data` → 容器`/assets/data`；`outputs/rlinf` → `/outputs` |

权重统一路径只读挂载，不为每个实验复制。环境变量见 [cluster.env.example](cluster.env.example)，
它是cluster/container配置，不应直接用于本地训练；默认WM权重和experiment已成对设为scratch rank32。
旧v1镜像仍需版本化LAM/chain及可选KIR/视频审计源码挂载，按原chain manifest选择；
不能覆盖原快照或叠加不兼容快照。GR00T的RL head/Flow-SDE实现仍在RLinf，不随上游包自动更新。

### 训练、评测和续跑

最优配方的提交预览（在cluster RLinf根目录运行，不提交作业）：

```bash
SWEEP_TAG=my_new_run \
bash docker/dreamdojo/submit_wm_comparison.sh --dry-run
```

确认后用新的唯一`SWEEP_TAG`，将`--dry-run`改成`--submit`。
`SEED_IDS=0/1/2`对应actor1234/1235/1236；脚本成对选择WM权重与匹配rank的experiment。
默认只产生最优配方这一组：scratch_r32、noise0.3、gb128/micro8、LR5e-6、
actor seed1234/env seed0、train15/eval35、KIR关闭；不是自动提交旧六组消融。
需要新对照时显式设置`WM_VARIANTS`、`NOISE_LEVELS`、`GLOBAL_BATCH_SIZES`、
`ACTOR_LRS`、`SEED_IDS`、`TRAIN_WM_STEPS`。r64配置已从DreamDojo精简，复用前先恢复匹配配置。
入口仍从SFT初始化，**不会默认加载第220代policy或恢复历史训练**。
上述是本地源码默认值；本次未同步cluster、改旧快照或重建SQSH。
更新后的`run_cluster.slurm`会显式传入配方，避免旧v1镜像中的YAML把默认值改回去。

独立eval仍用原生 `evaluations/eval_embodied_agent.py`，cluster包装为
`docker/dreamdojo/run_cluster.slurm eval`。指定 `runner.ckpt_path=<full_weights.pt>`；
SFT对照设为`null`。固定同一WM、eval35、KIR关闭、55唯一case，检查分母而非56槽位。
输出权重传容器路径，例如 `/outputs/1146382-train/...`，不是未挂载的host输出路径。

`train_chain.slurm`支持自动续跑：每段4h，`CHAIN_TIMEOUT=3.9h`、`MAX_RUNS=18`、
最长1000代；在完整代边界预留eval/保存时间。续训用 `runner.resume_dir=<完整global_step目录>`，
不是只加载policy权重。停止时在确认的chain目录放`STOP`，等待边界保存退出。
续跑恢复训练状态，但env/rollout随机流不是逐bit重放；新消融必须用新chain。

### 保留模型与运行检查

已做真机测试的220代权重：

```text
/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/1146382-train/wm_scratch_r32_n03_gb128_wm15_s1234/checkpoints/global_step_220/actor/model_state_dict/full_weights.pt
```

其他候选按上表job和实验名定位：`outputs/rlinf/<job>-train/<实验>/checkpoints/global_step_<代数>`。
cluster保留13份完整checkpoint，约264 GB；完整清单在输出根的
`.checkpoint_cleanup/20260917-retain13/completion.json`。旧latest指针可能指向已删除代数，不能直接恢复旧chain。

环境工具在 [docker/dreamdojo/](../../docker/dreamdojo/)：`prepare_context.py`准备构建上下文，
Dockerfile构建镜像，`export_sqsh.sh`/`upload_sqsh.sh`负责导出与上传。
`check_runtime.py`统一检查imports/CUDA/LAM/NCCL，先通过 `run_cluster.slurm smoke`和`verify`再训练。
这些是环境检查，不是模型效果评测；不要把token、数据或权重打进镜像。

本地入口：[run_dreamdojo_trocar.sh](../../examples/embodiment/run_dreamdojo_trocar.sh)；
配置：[YAML](../../examples/embodiment/config/dreamdojo_trocar_grpo_gr00t_n1d7.yaml)。
先加`--cfg job --resolve`查看配置。主YAML为8卡、64 env×2、global batch128；
本机使用shell入口仍避开故障GPU4，保留7卡，对应56 env×2、global batch112/micro8，
每代112条轨迹、20次更新。7不能整除128，因此本地只是硬件适配，不是完全相同的8卡复现；
严格复现用cluster的8卡设置。显式CLI覆盖最后生效；不要在本机直接使用未适配的8卡YAML启动。
定位问题用
[dreamdojo_validation.py](../../toolkits/world_model/dreamdojo_validation.py)，按parity → 真实动作WM → policy → reward顺序；
5/15/35指WM去噪迭代数，不是生成视频长度。

CPU回归统一在 [tests/unit_tests/dreamdojo/](../../tests/unit_tests/dreamdojo/)：
`env`（动作桥/reward/KIR）、`eval`（评测/视频/诊断）、`inference`（推理/LAM）、
`container`（打包/环境检查）、`chain`（续跑）、`sweep`（对比配置）六个测试文件。
在RLinf根目录运行，不加载GPU模型或提交cluster作业：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m pytest \
  --noconftest -p no:cacheprovider -c /dev/null -q tests/unit_tests/dreamdojo
```


当前仅保留220/SFT的独立eval视频，本地仅保留五组训练曲线，cluster训练日志仍保留。
DreamDojo当前仅保留v2分类器；v0/v1/v3产物和旧代码已归档到仓库外，可从
`/localhome/local-yunl/code_cleanup_archive/20260917_best_defaults_classifier.jp3dxhnS/`恢复。
其中130代原视频此前已删除，归档不能恢复这些视频；v2使用的`reviewed_labels_v3.json`仍保留，
文件中的v3是标签修订号，与被移除的四头v3模型无关。
旧文档及详细清理记录已归档到 `/localhome/local-yunl/code_cleanup_archive/20260917_single_doc.nUd2Rn/`；
此前直接删除的RL模型/视频没有备份，不能从文档归档恢复。本次不启动训练、不修改数据或正式权重。
