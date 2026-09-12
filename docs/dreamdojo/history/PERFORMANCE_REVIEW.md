# DreamDojo RL 耗时审查

> 历史性能检查；当前入口和配置见 [README](../README.md)。

日期：2026-09-10。本文保留实施优化之前的只读诊断与估算；其中“尚未优化／GPU 空闲”等描述对应当时状态。后续修改、同输入加速实测及多卡验收见 [DREAMDOJO_OPTIMIZATION_LOG.md](DREAMDOJO_OPTIMIZATION_LOG.md)，不要将下文历史估算当作优化后的实测。

## 结论与证据边界

主要瓶颈是 WM 轨迹生成，不是 GR00T 梯度更新。优先优化固定文本 embedding 的重复计算／模型搬运，以及 guidance=0 时没有跳过的无条件 DiT 前向；之后再比较减少去噪步数带来的速度和质量变化。

- **20 轮、35 步去噪的 trial 尚未启动**，没有这套新配置的每轮实测时间。当前 GPU 查询均为空闲。之前 6–10 小时只是预算估计。
- 训练计时来自已完成的两轮 smoke：单卡、8 个训练环境、group=4、每轮 20 chunks、WM 5 步、每轮评估 3 个起点并保存 checkpoint。
- 35 步计时来自固定 SFT 独立评估：WM 常驻 GPU，batch=8／9，没有优化器和 Ray，不能直接当作训练耗时。
- 完整 SFT 基线实际已完成：55 条，reward-model success_once=15/55=27.27%。[最终结果](logs/dreamdojo_review_20260910/sft_full55_35steps_seed0_retry01/results.json)。这不是实机成功率。
- 继续排除故障设备 GPU 4 的 UUID `GPU-77264942-9d19-703f-ff04-ee8e38ce7cb1`，不通过恢复使用该卡加速。

## 1. 已完成训练：每轮时间花在哪里

单位秒；百分比分母为两轮平均 `time/step`。依据 [TensorBoard 导出](logs/dreamdojo_review_20260910/grpo_report/results.json) 的 `grpo_smoke_v1.scalars`。

| 主流程阶段 | 第 1 轮 | 第 2 轮 | 平均 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| 生成训练轨迹 | 519.23 | 512.45 | 515.84 | 53.51% |
| GR00T 参数更新 | 36.88 | 33.47 | 35.18 | 3.65% |
| 评估（3 起点） | 375.09 | 381.90 | 378.49 | 39.26% |
| 训练前权重同步 | 7.85 | 1.32 | 4.59 | 0.48% |
| 优势／回报计算 | 4.10 | 0.01 | 2.05 | 0.21% |
| 未单列阶段的余量 | 30.34 | 25.34 | 27.84 | 2.89% |
| **整轮** | **973.49** | **954.49** | **963.99（16.07 分钟）** | **100%** |

余量是整轮减去前五项，不是单独测得的 checkpoint 时间。[Runner](rlinf/runners/embodied_runner.py) 的 `step` 包含 checkpoint 保存，但没有独立保存计时，不能把余量全认定为磁盘 IO。评估计时包含评估前权重更新。整次 smoke 含启动／退出为 [2157.44 秒](logs/dreamdojo_review_20260910/grpo_smoke_v1/timing.json)，比两轮 `time/step` 之和多 229.46 秒；这也不等于纯模型加载时间。

去掉每轮评估这一项，剩余平均 **585.49 秒＝9.76 分钟**，轨迹生成占 **88.10%**。这仍是 **5 步去噪** 的单卡 smoke。

轨迹生成内部：`time/env/env_interact_step` 平均 496.52 秒，占生成阶段约 96.25%，包含 WM、奖励和环境处理；`time/rollout/predict` 平均 8.98 秒，是已记录的策略预测时间。`actor/recv_traj` 约 516 秒主要是在等待轨迹，`rollout/generate` 也包含环境往返等待。**这些 worker 计时与主流程重叠，不能相加后说 actor／通信又各花了 8 分钟。**

## 2. 5 步 smoke：文本模型反复卸载是明显开销

从 [smoke 控制台日志](logs/dreamdojo_review_20260910/grpo_smoke_v1.console.log) 提取 80 次 latent generation，按 train20 → eval20 → train20 → eval20 区分。下表仅使用两组 train 的 40 次调用。

| 日志边界区间 | 每 chunk 平均 | 每轮 20 chunks 平均 |
| --- | ---: | ---: |
| 开始卸载文本模型 → 开始加载 VAE encoder | 8.675 秒 | 173.5 秒 |
| 开始加载 DiT → 开始 latent generation | 1.200 秒 | 24.0 秒 |
| 开始 latent generation → 开始卸载 DiT | 6.525 秒 | 130.5 秒 |
| 开始卸载 DiT → 开始加载 VAE decoder | 1.850 秒 | 37.0 秒 |
| 开始加载 VAE decoder → 开始卸载 decoder | 0.700 秒 | 14.0 秒 |

这是**秒级日志边界估计**，不是 CUDA profiler 的精确算子计时。文本卸载区间还包含同步／`empty_cache()`；latent generation 区间包含条件准备／VAE encode，不只有去噪。表格不覆盖全部时间，剩余包括文本编码及回搬、decoder 最后卸载、奖励、桥接、视频处理等，尚未逐项分离。VAE encoder 加载的相邻日志常处在同一秒，不能说它绝对没有成本。

仅文本卸载区间约 **2.9 分钟／轮**，相当于不含评估时间的约 29.6%。这不是缓存后已验证能省下的时间，但足以将其列为高优先级。

### P1：缓存固定文本 embedding

实际路径：

1. [RLinf 环境](rlinf/envs/world_model/world_model_dreamdojo_env.py) 每个 chunk 传 `prompt=[""] * B`、`guidance=0`。
2. [Video2WorldInference._get_data_batch_input](../DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py) 每次重新计算正／负文本 embedding，没有输出 embedding 缓存。初始化的 `self.neg_t5_embeddings` 没有在该路径发挥缓存作用。
3. [TextEncoder.compute_text_embeddings_online](../DreamDojo/cosmos_predict2/_src/predict2/text_encoders/text_encoder.py) 每次调用 `.model.to(self.device)` 后执行文本模型前向。
4. 训练 `enable_offload=true` 又让 pipeline 每次将文本模型搬回 CPU，固定提示词因此反复触发相同计算和搬运。

建议以模型版本、提示词、dtype、embedding 配置和 batch 形状为键缓存输出；先保持相同 batch 的计算协议，预热一次后让文本模型留在 CPU。**不是改用零文本条件，也不是缓存随状态变化的视频 latent。** 检查 embedding／生成结果和随机数消耗没有意外变化后再用于训练。

### P2：guidance=0 时跳过无条件 DiT 前向

实际 LoRA 路径为 `generate_vid2world` → `generate_samples_from_batch_lora` → `ActionVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch`。闭包目前始终执行：

```python
cond_v = self.denoise(noise, noise_x, timestep, condition)
uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
velocity_pred = cond_v + guidance * (cond_v - uncond_v)
```

来源：[动作条件模型](../DreamDojo/cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py)、[LoRA 采样器](../DreamDojo/cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py)。当前 guidance=0，有限数值下结果就是 `cond_v`。35 次采样调用 70 次 `denoise`，可考虑减到 35 次，**不改变去噪步数或动作条件**。

本轮用 AST 提取实际源码闭包，以标量 stand-in 执行：guidance=0、条件输出 3、无条件输出 17，调用顺序确为 conditional／unconditional，结果为 3。源码 SHA256：`ad63b9bd34ad897a60d2d8450bedef2cbc4bc941ad01a601cece169b85c88554`。这是控制流验证，**不是实际 GPU 输出 parity 或加速实测**。

建议仅对 guidance=0 加快速分支；检查非零 guidance 保持原行为、有限值、模型状态／随机数副作用及真实输出误差。理想收益是 DiT 前向工作量约减半，不代表整轮必然快 2 倍。

## 3. 35 步独立评估：去噪占稳定 chunk 时间约 91%

以下为重启后两个补跑分片，关闭 WM 内部 offload。前两个 chunk 排除在稳定态统计外。

| 分片 | batch | 稳定态去噪循环 | 相邻生成起点间隔 | 分片总时长（含加载、输出） |
| --- | ---: | ---: | ---: | ---: |
| rollout_30k_00 | 9 | 约 52 秒 | 57.35 秒 | 1245.76 秒＝20.76 分钟 |
| rollout_30k_01 | 8 | 约 46 秒 | 50.18 秒 | 1099.87 秒＝18.33 分钟 |

依据：[batch9 日志](logs/dreamdojo_review_20260910/sft_full55_35steps_seed0_retry01/rollout_30k_00.console.log)、[batch8 日志](logs/dreamdojo_review_20260910/sft_full55_35steps_seed0_retry01/rollout_30k_01.console.log)。每份 20 次去噪有两条重复的最终 `100%` 行，统计时每对只算一次，不能当作 40 次调用。进度条时长也是整数秒近似。

去掉 offload 后，35 步 DiT 成为绝对主项。每个 chunk 推进机器人 12/30＝0.4 秒，20 个串行 chunk 才得到 8 秒闭环轨迹。增加 GPU 可并行更多轨迹，但**不直接缩短同一轨迹的 20 次前后依赖**。新 trial 的 40 环境分到 5 卡后仍是每卡约 8 环境，不能把单卡 batch8 的时间再除以 5。

### P3：WM 在整段 rollout 内常驻，阶段切换时再卸载

[环境构造](rlinf/envs/world_model/world_model_dreamdojo_env.py) 将一个 `enable_offload` 同时用于 DiT、文本模型和 VAE；pipeline 的搬运发生在**每个 chunk 内**，与外层 worker 阶段性卸载不是一回事。

建议先做文本缓存，再分别测试 DiT／VAE 常驻。独立 batch8／9 评估已有不卸载的成功运行，但不证明 actor、rollout、FSDP 和两个 env 实例共存的训练也装得下。日志约 30–31 GiB 只是构建 data_batch 后的单进程 allocated memory，**不是推理峰值或整卡占用**。需要测 peak allocated／reserved、整卡显存及训练阶段峰值，不能直接关闭所有组件 offload。

### P4：评估调度影响 trial 总完成时间

旧 smoke 每轮评估占 39.26%；待启动 trial 草稿已经改为每 5 轮在另外两卡评估，不应继续按每轮同步评估估计。

但 full55 有 7 个固定分片，已完成分片累计约 **130.42 GPU·分钟**。按草稿交替分配到两卡的队列，既有分片时间相加约 **73.53／56.89 分钟**；这是历史时长外推，不是新 checkpoint 实测。两卡 full55 不能宣称 20 分钟完成。评估可与训练重叠，不应将四次评估简单全部加到训练上；仍需计入最后评估尾部和可能的排队。

可优化：评估进程常驻，依次切 checkpoint／分片以减少重复加载；每 5 轮固定小集合筛查，关键 checkpoint 再完整 full55；或调整训练／评估卡数。**频率、覆盖范围变化须先确认**。checkpoint 对照保持既定 batch、种子和 episode 次序，不为并行度拆小 batch 后仍声称完全同协议。

### P5：降低去噪步数，需要质量对照

在 P1／P2 后测试 35／15／10／5 步。它直接减少 DiT 工作量，但固定成本仍在，35→5 不等于整轮快 7 倍。

已有三条 GT 回放中 5／35 均完成三阶段，但策略闭环轨迹会随步数变化，三起点不足以证明全量等价。保留 35 步作为质量参考，对照视频运动／清晰度、三阶段概率、动作和同起点成功率后再决定训练步数。

## 4. 暂不优先的优化

- GR00T 反向、FSDP、LoRA rank、optimizer：旧 smoke 参数更新仅占整轮 3.65%，不是首要瓶颈；仍不能跳过新五卡通信验证。
- 奖励：逐帧双次 30Hz 更新有 CPU resize、GPU 往返和标量同步，可测后考虑批处理；不能直接删除双次更新，时序窗口／ratchet 会改变。没有独立计时，不编造占比。
- 视频：可减少录像数量、异步编码，但保留调试证据。没有独立计时前不把 27.84 秒余量全部归因于视频。
- 原生生成 13 帧、只消费 6 个未来帧：存在未消费未来，但 horizon、VAE 时域压缩、动作条件与训练分布耦合。不能直接裁成 7 帧就声称等价省一半计算。

## 5. 下一步验证顺序（尚未执行）

1. 固定 SFT／WM checkpoint、batch8、I2 起点、输入／种子及 35 步，在非 GPU4 设备做有预热的短 profile。分别量文本、H2D／D2H、VAE encode、两路 DiT、decode、reward、policy、视频与显存峰值；用同步边界计时，区分初始化和稳定态。
2. 分别验证文本缓存和 guidance=0 快速分支：每项先检查输出 parity，再跑 20 chunks 闭环，报告端到端时间，不只报算子收益。
3. 真实训练共置方式下验证 WM 常驻显存及至少两轮稳定运行；五卡测最慢 rank，不能用单卡均值替代关键路径。
4. 再比较去噪步数，确定配置后从原始 SFT 启动 trial；不用旧 step20。

启动前另有阻断项：草稿 `dreamdojo_trial.py::read_health()` 读取 `output/EXPERIMENT/tensorboard`，但 `MetricLogger` 写 `output/tensorboard`；该 trial 开启 `per_worker_log` 后汇总目录为 `output/tensorboard/all`。草稿会误报 initializing、漏查非有限指标，需修正并测试后才能可靠监控。这不是旧 smoke 慢的原因。本轮仅登记，未修改脚本，也未把上一轮中断的测试当作通过。

本报告是诊断和实验建议，不是“优化已实施／性能已提升”的验收记录。
