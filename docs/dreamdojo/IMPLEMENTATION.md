# 实现契约与已修复问题

本文件记录需要长期保留的实现知识。时间线和一次性命令已归档，见 [CLEANUP.md](CLEANUP.md)。

## 1. Pipeline 与参数更新边界

```text
真实 teleop ──→ GR00T N1.7 SFT ──→ 不同SFT checkpoint真机rollout
      └───────────────────────────────────┤
                                         ↓
                           teleop + 成功/失败rollout
                           ├→ DreamDojo任务适配（冻结WM）
                           └→ 人工阶段标注 → v2 classifier（冻结reward）

SFT初始化的GR00T → 动作桥 → DreamDojo生成下一观测 → classifier奖励
       ↑                         │
       └────── GRPO更新 ──────────┘
       ↓
RL checkpoint 与保留SFT在真机对照评测
```

WM / classifier 不参与 GRPO 参数更新。GR00T 的 RL action-head、Flow-SDE 和 log-prob
实现保留在 RLinf；相邻 Isaac-GR00T 提供模型/processor等依赖，不能只替换 upstream包就认为
RL head也同步改变了。当前依赖源版本为 `2d9a9fc`；此次代码整理没有再次升级它。

## 2. State / action / 时间对齐

核心：[dreamdojo_adapters.py](../../rlinf/envs/world_model/dreamdojo_adapters.py)、
[lerobot_world_model.py](../../rlinf/data/datasets/lerobot_world_model.py)、
[world_model_dreamdojo_env.py](../../rlinf/envs/world_model/world_model_dreamdojo_env.py)。

- Policy / RL 数据接口是原始28D；DreamDojo训练的数据经过G1填充适配为43D，
  然后按统计量、时间采样与block delta规则构成384D conditioning。
- 手臂字段是绝对目标；绝对 arm state 重复累加的问题已在本轮最早检查前修复。
  后续不能把 absolute target 再加到 previous state。
- 保留原始物理单位，不对已解码的 physical action 机械执行 `[-1,1]` clamp。
- 正式 bridge 对齐两倍时间采样、4-step block delta 与 `state_t` 历史前缀。
  默认起点frame2，需frame0历史；KIR非零起点也必须重建一致历史。
- 30Hz真实命令 / 15HzWM；12条policy动作产生6张新WM帧。
  8秒轨迹通常保存初始帧加120张新帧，共121帧。

**Parity**：给两条实现完全相同的输入，比较中间张量的数值/排列/时间索引，
而不是只比较视频“看起来差不多”。已做真实 recorded-action 与 DreamDojo dataloader 的384D
conditioning核对；保留CPU适配器测试。RL应读取28D原始数据，不应换成WM适配后的43D数据目录。

## 3. WM 定位方式与性能

`toolkits/world_model/dreamdojo_validation.py` 保留以下诊断，不是替代原生eval：

| 模式 | 作用 / 不能证明什么 |
| --- | --- |
| parity | 检查bridge与数据处理数值一致性 |
| recorded teacher forcing | 每块从真实图像/历史开始，未来用真实动作；检测局部生成能力 |
| recorded closed loop | 下一块接上块生成图；分真实历史和在线proprio proxy两种，分离图像累积误差与状态误差 |
| policy | 在同一起点加入SFT/RL策略，定位policy/state/action接口问题 |
| reward / report | 检查生成域概率、轨迹和数值，不能把classifier输出当人工真值 |
| kir | 检查非零reset、组内共享起点、后缀奖励与动作桥 |

真实动作小测证明管线能产生任务进展，不证明WM长期物理正确。
KIR修复后3条训练演示×8样本的真实动作模式均被classifier判为完成，
但只有3个相关case，不能当24条独立测试或55例完整任务eval。

已保留的优化：多env批量WM、批量reward、文本embedding缓存、guidance=0快速路径、
模型驻留/卸载边界与VAE缓存。代码中的Wan VAE是Cosmos使用的tokenizer实现，
不表示本项目改用了Wan世界模型。Actor micro batch与WM推理batch无直接绑定关系。

## 4. 必须防止复发的问题

| 问题 / 修复 | 保留的验证 |
| --- | --- |
| absolute state重复累加、物理动作错误裁剪、conditioning时间/维度错位 | adapter测试与parity诊断 |
| LAM ckpt缺失/不匹配时静默跳过 | 严格加载，失败直接抛错；LAM与container测试 |
| unique eval按槽位重复计数 | 55唯一case；padding不计入success分母 |
| GR00T `eval()` 返回None | 保留model对象，再单独调用to/eval；diagnostics测试 |
| 组内共享图像遭原地Normalize重复修改 | 每env clone；真实Normalize回归检查范围和8份初帧一致 |
| embedding缓存被调用方修改、dtype/config过期 | 精确数值、LRU、dtype/config及train-mode失效测试 |
| 不完整checkpoint被续跑读取、丢失global step | 原子完成标记、分片校验、native runner预算边界测试 |

严格LAM加载已在cluster8 ranks验证。需特别区分：
当前机器人动作直接编码成384D，rollout不依赖LAM重新抽取latent action；
“LAM成功加载”不等于“LAM正在决定每次robot action conditioning”。

GR00T升级后 `letterbox=false` 被正确执行，观察到256×340而非先前256×256预处理。
Flow-SDE/log-prob逻辑未因此改写，但输入处理变化可能影响policy行为；版本比较必须固定processor。

## 5. KIR：只改变训练初始状态

实现基于人工阶段标签选取 handover 前、pick 已完成且历史可用的帧，
不是把每条数据前10/30帧统一叫关键帧。offset15/30指相对事件向前选帧的最大偏移，
单位为原始30fps帧；不是额外视频时长。

具体选帧为 `handover - min(max_offset, (handover - pickup) // 2)`；
要求成功标注、三个事件有效且有序、远离pick/handover各自±5帧模糊边界，
并有至少16帧过去历史；不满足则不做该case的KIR。

- 按GRPO组抽是否启用KIR；同组8个env用同一个episode/起始帧。
- 同步reset图像、proprio、动作历史和classifier的过去历史；不能只换一张图片。
- 起点已完成pick时初始化stage=1，但累计return=0，不奖励起点已有成绩。
- 无合法KIR候选则回退正常起点。eval始终关闭KIR，仍从原始起点完成整个任务。
- v1/v2诊断存在共享图像污染，已作废；v3修复后才开始正式KIR长训。
- 功能正确不等于收益确定；多seed结果不足以支持默认开启，详见RESULTS。

## 6. Reward 与视频审计的边界

`dreamdojo_reward.py` 使用v2三头概率，阈值0.8；
pick/hand要求15个tick中13个，place要求2/2。
模型真实历史偏移0/4/8/16@30fps；15fps生成帧重复两次送入reward。
因此place的两个tick可能来自同一生成图，不是两个独立新观测。

当前reward是历史stage单调锁存0→3，每新增阶段+1，总分最多3。
掉落后classifier当前标签可回退，但已经领取的reward不会撤销；
同一时刻多个头确认可能跳阶段。错误pick可能放行后续奖励。

视频上的预测、历史stage和人工任务成功必须分开。
`dreamdojo_video_audit.py` 保存逐帧概率、stage及预测覆盖层，方便核查，
不产生人工真值。左手盖右手可能是policy固定模式，也可能受WM遮挡偏差强化；
需要实际动作trace与真机复核，不能只看视频下因果结论。

v3只完成稀疏AI标注和候选训练，未进入正式27组RL。
其有效性头测试仅检出3/10异常；不能直接换checkpoint路径上线。
详见 [分类器主文档](../../../DreamDojo/docs/MILESTONE_REWARD.md)。

## 7. 续跑的可复现范围

保存policy、optimizer/scheduler、actor RNG与完成代数，严格检查DCP各rank分片。
Slurm边界使用预算预测和原子checkpoint完成标记，非强行在3.9h中断写入。
env / rollout RNG没有完整持久化，所以恢复是训练状态续接，不是逐bit重放未来轨迹。
独立eval与续跑必须记录WM/config/source版本；不要覆盖旧chain依赖的版本化运行源码。
