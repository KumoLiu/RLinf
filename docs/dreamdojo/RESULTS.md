# 实验结果与保留模型

截至 2026-09-16；集群结果来自 2026-09-15 最终审计，没有在整理文档时重新运行 GPU 评测。

## 1. 对外报告的主结果

| 评测环境 | 模型 | 成功率 | 证据性质 |
| --- | --- | ---: | --- |
| 真机，大致相同条件 | 任务 SFT base | 44% | 用户报告 |
| 真机，大致相同条件 | RL 第 220 代 | 66% | 用户报告；+22 个百分点 |
| DreamDojo + v2 classifier，55 例独立视频 eval | SFT base | 25/55 = 45.45% | 分类器判定，不是真机结果 |
| 同上 | 同一实验第 130 代 | 38/55 = 69.09% | job 1138339 |
| 同上 | 同一实验第 220 代 | 41/55 = 74.55% | job 1149841 |

SFT 视频 eval 为 job 1147182。130 与 220 都属于
`wm_scratch_r32_n03_gb128_wm15_s1234`，chain 1120517，**没有 KIR**。
原生训练内对应 eval 为 40/55、43/55；独立 eval 的随机动作 latent / WM 轨迹不同，
不应要求与训练内逐条相同。所有这些 WM case 已反复用于模型选择，不是新的盲测集。

真机还缺试验总数、逐次成功记录、统一判定准则及条件匹配记录。
当前可以报告观察到的提升，不能从两个百分比推出统计显著性。
分类器可能误判 pick/hand/place，不能用 WM 高分替代肉眼或真机验证。

### 已完成真机测试的权重

```text
/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/1146382-train/wm_scratch_r32_n03_gb128_wm15_s1234/checkpoints/global_step_220/actor/model_state_dict/full_weights.pt
```

## 2. 实际训练设置

| 项目 | 最优已测试单模型的设置 |
| --- | --- |
| Policy | 任务 SFT GR00T N1.7 初始化，GRPO 更新其可训练参数；policy 去噪 4 步 |
| 冻结 neural simulator | DreamDojo 2B，scratch LoRA rank32 / WM LR3e-4，iter18000 EMA |
| 冻结 reward | v2，pick / handover / place 三头 |
| 采样 | 64 个并行 env × 2 rollout epoch = 128 条轨迹/代 |
| GRPO | group8，共 16 组；同组共享 case/起点，随机采样不同 |
| 时长 | 240 条 30Hz 命令，8 秒；12 条/动作块，共 20 块；WM 视频 15fps |
| Actor | LR5e-6，global batch128，micro batch8，noise0.3；physical action noise0，不退火 |
| 世界模型去噪 | train15 / eval35 |
| 初始化 / 评测 | train KIR关闭；eval 始终 KIR关闭，55 唯一 case，56 槽位中的 padding 不计分 |
| 周期 | 每5代 eval / 保存；max_epochs1000；每段4h，CHAIN_TIMEOUT3.9h，MAX_RUNS18 |

一代有 128 × 20 = 2,560 个动作块，global batch128 对应 20 次 actor 更新。
global batch256/512 分别为 10/5 次；micro batch 控制每卡一次前反向的大小，
不控制 WM 每次推理的 env 数。不同全局 batch 的比较同时改变每代更新次数。

SFT 仅作初始化和独立对照；本轮 kl_beta=0，没有 SFT co-training。
训练中的 approximate KL 对应旧 rollout policy，不是固定 SFT reference。

## 3. WM 验证集候选（不是新的真机最佳）

以下均在 scratch WM35 + v2 + 同55例协议下评测。单点峰值有验证集择优偏差；
最近三次均值的训练预算不同，也不能直接当公平排名。

| 候选 | 训练配方（均 noise0.3） | checkpoint | 原生 eval 峰值 | 各自最后三次均值 |
| --- | --- | ---: | ---: | ---: |
| C1 | WM15 / gb128 / seed1234 / 无KIR | 320 | 46/55 = 83.64% | 76.97% |
| C2 | WM35 / gb128 / seed1235 / 无KIR | 190 | 41/55 = 74.55% | 53.33% |
| C3 | WM35 / gb128 / seed1234 / KIR p0.5 offset15 | 115 | 40/55 = 72.73% | 66.67% |
| C4 | WM35 / gb256 / seed1234 / 无KIR | 175（165同分） | 40/55 = 72.73% | 53.94% |
| C5 | WM15 / gb128 / seed1234 / KIR p0.5 offset15 | 145 | 39/55 = 70.91% | 56.97% |

C1 在185、200代也达到46/55；315/320/325的均值80%。
**320 代尚无新的独立视频 eval 或真机结果，不能覆盖220代的实测结论。**

以下路径相对 cluster 根 `/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/rlinf/`：

```text
C1: 1177445-train/wm_scratch_r32_n03_gb128_wm15_s1234/checkpoints/global_step_320
C2: 1178228-train/wm_scratch_r32_n03_gb128_s1235/checkpoints/global_step_190
C3: 1151683-train/wm_scratch_r32_n03_gb128_kirhandp50_off15_s1234/checkpoints/global_step_115
C4: 1143729-train/wm_scratch_r32_n03_gb256_s1234/checkpoints/global_step_175
C5: 1173304-train/wm_scratch_r32_n03_gb128_wm15_kirhandp50_off15_s1234/checkpoints/global_step_145
```

各目录的独立 policy 文件为 `actor/model_state_dict/full_weights.pt`。
续训需要整个目录的训练状态，不是只加载这一个文件。
原 WM 的候选 `wm_r64_n01_gb128_s1234` 在165代为36/55；因模拟环境不同，不插入上述排名。

## 4. 对比实验得到什么结论

- **已证实的工程收益**：原生训练/评测、批量 WM、严格 LAM 加载、自动续跑可用；
  WM15 主线相对 WM35 显著省时。
- **最强单模型，不是稳健最优配方**：WM15 / noise0.3 / gb128 / seed1234 表现最好，
  但 seed1235/1236 没有稳定复现，不能只看最高的一条曲线。
- **Noise**：共同30/35/40代，scratch WM35 的 noise0.1/0.2/0.3/0.5 均值
  为47.88/51.52/56.36/40.00%；支持0.3作为候选，不支持无限增大探索。
  这是训练 Flow-SDE noise，不是给未来视频直接加不同物理动作噪声。
- **Actor LR / batch**：1e-5 早期较差；2.5e-6 有时缓解回落但未超过主线。
  gb256 在 WM35 有可用候选，在 WM15 未稳定改善；gb512 没有足够优势。
- **KIR**：p0.5 / offset15 的 seed1234 有潜力；其余 seed 不支持稳健收益。
  p1 或 offset30 没有一致优势，故不建议默认开启。
- **两个 WM 谁更好**：各自在自己的模拟器中训练和评分，存在“环境更容易/更容易骗 reward”的混淆。
  需固定同一 policy 做交叉评测、recorded-action 回放及真机测试，不能只比较对角线 success。
- **剩余瓶颈**：WM 接触/遮挡时的物体形变、分类器误报、reward 历史锁存，
  都可能鼓励固定的遮挡/交接动作。仅靠视频不能区分 policy 模式收敛和 WM 偏差。

多 seed 的共同窗口，先每 seed 平均三次 eval，再跨 seed 平均：

| 完成代数 | WM35 无KIR | WM15 无KIR | WM35 KIR p0.5 offset15 |
| --- | ---: | ---: | ---: |
| 100/105/110 | 65.05% | 36.57% | 43.43% |
| 130/135/140 | 41.21% | 49.29% | 未完成完整三seed共同窗口 |

结论随阶段改变；不能用主线338代对比补充seed188/190代来宣布算法优势。

## 5. 训练一代多久

主线338代的 TensorBoard 统计，8卡 cluster；不含排队、启动和部分续跑边界保存：

| 部分 | 平均耗时 |
| --- | ---: |
| 采样 / WM rollout | 495.84秒，约8.3分钟 |
| Actor 更新 | 约65秒，约1.1分钟 |
| 普通代（无eval） | 565.36秒，约9.4分钟 |
| 带eval/保存的代 | 1026.71秒，约17.1分钟 |
| 单次eval | 427.13秒，约7.1分钟 |
| 含每5代eval的摊销平均 | 656.81秒，约10.95分钟 |

前220代累计 `time/step` 约40.18小时。WM35/gb128 对照摊销平均约18.1分钟/代，
WM15约10.95分钟，降低约39.5%。这些是实际日志统计，不是每种 seed 的固定 SLA。

## 6. 完整证据与停止状态

最终审计：27组正式实验、302个日志段、3,898代、772次原生eval，分母全部为55。
原生 TensorBoard 的 step 从0开始：本文完成代数 = TB step + 1。

- [27组全量总表](../../logs/final_review_20260915/experiments.csv)。
- [逐实验/窗口汇总](../../logs/final_review_20260915/summary.json)、
  [原始标量整理](../../logs/final_review_20260915/metrics.json)。
- [Checkpoint结构审计](../../logs/final_review_20260915/checkpoint_audit.json)：
  65个目录的global_step、full_weights、DCP元数据及8分片已检查，不等于全部GPU重载过。
- [可比TensorBoard目录](../../logs/final_review_20260915/tensorboard/comparison/)：
  27条唯一曲线可在多个分组重复展示，52个展示条目不是52组实验。
- 原始视频/评审仍在 `logs/best_ckpt_eval_20260913/`、
  `logs/eval130_200_review_20260914/`、`logs/reward_motion_review_20260914/`。

2026-09-15 15:32 UTC 的最后审计确认项目集群任务已全部停止。
若 Slurm 末段为 `MAX_RUNS=18 reached; checkpoint retained`，属于续跑上限退出，
不是 CUDA 失败，也不是达到1000代。主线最终保存338代。
此处是历史状态，不是2026-09-16实时队列检查；本轮整理不提交/取消任何作业。

## 7. 下一阶段优先级

1. 保存220代真机逐次结果；对候选320代与同一SFT做条件匹配、独立人工判定的复评。
2. 先修复 reward 的困难负例覆盖与掉落语义，再做大规模新 sweep；v3尚不宜替换v2。
3. 新实验固定预算、多个seed、独立case；WM选择使用交叉评测，不按各自reward单点挑选。
4. 方法扩展：deformable任务、multi-camera；Cosmos Simulator / Cosmos3属于计划，
   当前代码与结果仍是 DreamDojo / Cosmos Predict2.5，未做该迁移。
