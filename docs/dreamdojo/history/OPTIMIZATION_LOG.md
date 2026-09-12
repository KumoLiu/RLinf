# DreamDojo 推理优化实施与验收

> 历史优化验收；以下旧命令可能已退役，当前用法见 [README](../README.md)。

日期：2026-09-10。诊断依据：[性能审查](DREAMDOJO_PERFORMANCE_REVIEW.md)。本轮按用户要求实施优化，保持原始 SFT、DreamDojo 权重、35 次去噪、动作／状态桥接和评测协议不变。旧 step20 不参与。GPU 4 的 UUID 始终排除。

最终状态：同输入 WM 微基准、20-chunk SFT 闭环对照、五卡两轮真实训练均正常完成。已启用通过验收的优化；20 轮 trial 没有启动。本次验证性能及数值一致性，不宣称 RL 成功率提升。

## Batch inference 约定

WM 输入图像为 `[B,3,13,480,640]`，动作条件沿同一 batch 维堆叠。一个 worker 的多个 env 一次调用 DiT 生成，不是逐个 env 各跑一遍 DiT。GRPO 同组起点相同，但动作、WM 采样及奖励状态仍分别维护。每条轨迹的 20 个 chunk 前后依赖，不能把时间维当作互不依赖的 batch。

两轮旧 smoke 为单卡 batch8；35 步全量 SFT 分片 batch 为 9／8／7／6；20 轮 trial 草稿为 5 卡共 40 train env，即每卡 batch8。增加卡数主要提升同时采集的轨迹数量，不能把同一 batch8 的延迟再除以 5。

## 追溯与回退

- 修改前快照：`logs/dreamdojo_review_20260910/perf_before_v1/`，包含 RLinf、DreamDojo、GR00T 当时已有未提交差异，未覆盖用户已有改动。
- 最终快照：`logs/dreamdojo_review_20260910/perf_after_v1/`；`changes_since_before.patch` 只列相对本轮 before 的变化，`manifest.json` 保存源码 SHA256。各 GPU 实验还保留当次启动的独立源码快照；后续测试脚本／文档修改不会回写这些快照。
- 新诊断入口：`toolkits/world_model/dreamdojo_perf.py`，复用现有 LeRobot 加载、RLinf DreamDojoEnv 和视频／奖励导出；输出目录必须不存在。每次运行保存源码快照、参数、结果及异常记录。
- 优化可独立关闭：`cache_text_embeddings: false`、`skip_zero_guidance: false`；三个 `wm_offload_*` 设为 `null` 时继承旧 `enable_offload` 行为。
- DreamDojo 通用 pipeline 新开关默认关闭，仅在通过验证后的 RLinf recipe 中选择启用。换文本编码器权重时必须调用 `clear_text_embedding_cache()`；只换 GR00T 或 WM DiT LoRA 不改变冻结文本编码器。
- 未提交 Git；不覆盖权重、已有视频或数据集。最终启用状态和验收结果应以下面的完成记录为准。

## 已实施的代码变化

1. **固定文本缓存**：新增 `TextEmbeddingCache`，LRU 最多 4 个整 batch 条目，保留原 dtype，缓存 CPU 副本；命中返回独立 tensor，避免 conditioner 修改污染缓存。键包含完整 prompt 顺序／batch、pipeline checkpoint、编码器身份／配置、权重 dtype、CUDA autocast 状态。编码器 training 模式不使用缓存。保持原空提示词／负提示词，不改成零张量，不缓存图像 latent。
2. **零 guidance 单分支**：动作条件 rectified-flow 在显式开关开启且 guidance=0 时仅算条件 DiT；保留原条件构造和非零 guidance 行为。35 次采样的 `denoise` 次数由 70 减至 35。真实输出和随机状态是否完全一致由 GPU 对照验收，不仅依据公式推断。
3. **分离 chunk 内与 rollout 边界的显存管理**：允许 DiT／VAE 在一整段 rollout 内常驻，外层 `offload()` 时迁回 CPU、`onload()` 时恢复；文本通过缓存只需预热后卸载。缺省空值保留旧行为，常驻需通过训练共置显存测试再选用。
4. **启动草稿修复**：修正 TensorBoard 健康检查路径为 `output/tensorboard/all`，训练报告兼容这个路径及旧 `output/tensorboard`；trial YAML 改为先 compose 主 recipe、再合并显式 overlay，避免继承带 `hydra.searchpath` 的配置时报错；修复独立 full-eval 的模块导入路径。未改变 full55 覆盖、分片顺序或每 5 轮的检查频率。

## Wan VAE 接口澄清及失败记录

DreamDojo 本来就使用 Wan2.1 VAE 做像素／latent 编解码，预测未来的仍是本次训练的 DreamDojo 动作条件模型；与 RLinf 的 WanEnv 不是一回事。本次没有更换 VAE 或模型架构。

`perf_micro35_v1` 在诊断器初始化时失败：实际 `Wan2pt1VAEInterface` 没有通用代码假设的顶层 `.encoder/.decoder`，共享网络是 `.model.model`。已修正诊断器和新增的阶段迁移逻辑，并加入 CPU 回归；原失败 `error.txt` 保留。VAE 的运行时特征缓存在整段 rollout 切换时清空。

这也限定了上一轮性能报告：pipeline 的 VAE loading/offloading 日志外有属性判断，Wan VAE 路径可能只打印而没有实际迁移，不能据日志文字推断真实 VAE 搬运量。文本和 DiT 的冗余判断不受此影响。新验收以同步后的实际模块调用计时为准。

## 验证结果与边界

### 同输入微基准：完成

目录：[perf_micro35_v2](logs/dreamdojo_review_20260910/perf_micro35_v2/results.json)。GPU 1，teleop validation 起点 0–7，batch8，固定真人动作的首个 chunk，seed 0，35 步去噪。每个方案独立 reset、1 次预热、2 次稳态测量。以下延迟包含 WM、reward 和环境处理，不包含首次模型加载或输出文件写入。

| 方案 | 稳态秒／chunk | 相对 baseline 加速 | DiT 调用／chunk | 预热后文本前向／chunk |
| --- | ---: | ---: | ---: | ---: |
| 原训练卸载方式 baseline | 64.28 | 1.00× | 70 | 2 |
| 仅固定文本缓存 | 51.30 | 1.25× | 70 | 0 |
| 缓存＋零 guidance 单分支 | 28.95 | 2.22× | 35 | 0 |
| 再加 rollout 内常驻 | 25.75 | 2.50× | 35 | 0 |

全部方案及重复的浮点生成帧、奖励、三阶段概率、正负文本 embedding、CPU／CUDA RNG 状态均与第一份 baseline **逐元素相同，最大误差 0**。浮点帧比较覆盖交给环境的生成输出，不只是压缩后的视频。`reference.pt` 保存基准张量，`results.json` 保存每次检查和计时。没有降低 35 次去噪、分辨率或 batch。

resident 的一个稳态样本：DiT 21.76 秒，VAE encode 1.11 秒、decode 1.99 秒，reward 0.28 秒，输入／文本 0.50 秒，整 chunk 25.74 秒。计时为 CUDA 同步后的 inclusive 时间，嵌套项不能相加。诊断进程峰值 allocated 约 65.56 GiB；最高 reserved 78.13 GiB，包含诊断张量引用，不能代替多进程整卡峰值。每个方案只有两次稳态测量，不给出统计显著性或全训练 2.5× 的结论。

### CPU 与通信

- 首次 45 项中 44 通过，1 项暴露 trial Hydra 继承问题；修正后 45/45（11.26 秒），补入 Wan VAE 测试后 46/46（10.34 秒）。加入诊断初始化与报告路径回归后 49/49（14.46 秒），结果为 `perf_tests_v1.xml`；最终启用配置后 **49/49（10.79 秒）**，结果为 [perf_tests_v2.xml](logs/dreamdojo_review_20260910/perf_tests_v2.xml)。
- `perf_collective5_v1/rank_0.json` 至 `rank_4.json`：物理 GPU 0／1／2／3／5 的五 rank NCCL all-reduce 和 all-gather 全通过，沿用 `NCCL_P2P_DISABLE=1`。GPU 4 不参与计算。
- 修改的 RLinf Python 文件 Ruff lint 通过，诊断入口及测试格式检查通过。DreamDojo 两个既有大文件存在修改前就有的 lint 报告；对照 before 未引入新规则／消息类别，未为清理旧问题而重写整个文件。

### SFT 完整闭环对照：完成

首次 `perf_policy35_v1` 在首个动作预测前失败：测试脚本使用 `model = get_model(...).to("cuda").eval()`，而当前 GR00T 的 `eval()` 返回 None。已把构建、to、eval 拆成独立调用，并加入真实初始化语句的 CPU 回归；不修改 GR00T 模型实现或权重。失败 `error.txt` 保留。

重试目录 [perf_policy35_v2](logs/dreamdojo_review_20260910/perf_policy35_v2/results.json)，退出码 0、总时长 721.42 秒。GPU 7，teleop validation 起点 0／1／2、seed 0、35 步、20 chunks。对照旧独立评估方式 `baseline_resident` 与优化 `resident`；两组由原始 SFT 每步预测动作，之后只消费 WM 生成观测，无 teacher forcing。

完整轨迹的动作、压缩前 uint8 视频帧、浮点三阶段概率、奖励和 CPU／CUDA RNG 均最大误差 0。两组最终阶段均为 `[1,1,3]`，return `[1,1,3]`。排除首个预热 chunk，平均延迟 **18.85 → 10.18 秒，1.85× 加速**。allocated 峰值 50.41 → 34.97 GiB，reserved 55.74 → 38.53 GiB；这是 batch3 策略＋WM 诊断进程，不能与 batch8 微基准直接比显存。两组目录各包含三份 `episode_*.mp4`、`trace.npz` 和 `metrics.json`。

三起点用于数值等价检查，不替代固定 full55，也不代表新的成功率评估。WM 微基准用固定动作比较浮点生成输出；本闭环测试则覆盖完整反馈链路，不以“看起来差不多”代替数值检查。

### 真实 RLinf 五卡两轮：完成

目录 `perf_grpo35_5gpu_v1`，物理 GPU 0／1／2／3／5，40 个 train env、每卡 batch8、group8；35 步、20 chunks，从原始 SFT 初始化。保留三份原 train 数据的 0.34／0.33／0.33 混合；关闭内部 eval，只在第二轮保存 checkpoint。验证常驻 WM 在 rollout → actor update → 下一次 rollout 的边界卸载及多卡通信，不用于证明 RL 提升。

退出码 0，两轮含启动／结束总计 **1,804.00 秒＝30.07 分钟**。完整指标和 checkpoint 清单：[perf_grpo35_report_v1/results.json](logs/dreamdojo_review_20260910/perf_grpo35_report_v1/results.json)。TensorBoard step 0／1 对应 runner 第 1／2 轮。

| 指标 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 整轮秒数 | 799.87 | 728.54 |
| 生成训练轨迹 | 540.53 | 490.28 |
| actor update | 211.77 | 201.29 |
| 权重同步 | 47.49 | 8.96 |
| 优势／回报计算 | 0.078 | 0.016 |
| grad_norm | 0.3610 | 0.4645 |
| approx_kl | 0.000616 | 0.000215 |
| 训练 success_once | 1/40 | 3/40 |
| 平均训练 return | 0.700 | 0.825 |
| 组内奖励完全相同的比例 | 40% | 0% |

所有导出的标量有限。第二轮包含 checkpoint 保存，整轮减去四个主阶段的约 28 秒是未单列余量，不能全部认定为磁盘 IO。两轮主流程外约 276 秒包括初始化／结束。`perf_smoke/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt` 已保存，6,910,740,025 字节；仅为 smoke 产物，不替换 SFT 或自动作为长训起点。

训练采样起点与 full55 不同，两轮的起点也会重采样。因此不能将 2.5%／7.5% 与原 full55 27.27% 当作退步，也不能把两轮间变化当作已验证的 RL 提升。group 指标证明这两批并非全部零奖励／零优势。

初始化期间部分只读 GPU 查询超时，后来五卡正常计算；检查对应时段内核日志未见新 Xid，不能据查询慢声称又坏了一张卡。`gpu_samples.jsonl` 共 253 条，其中 15 次查询超时；训练卡最高观测占用 66,727 MiB（约 65.16 GiB），GPU 4 为 8 MiB／0% 且未分配训练 worker。采样间隔约 5–7 秒，**观测最大值不是精确峰值**；不能据此直接翻倍 batch。训练正常完成，未见 OOM。

### 最终启用配置与剩余工作

- `env/dreamdojo_trocar.yaml`：`cache_text_embeddings=true`、`skip_zero_guidance=true`、`wm_offload_text_encoder=true`。DiT／VAE 的 split offload 仍为 null，保留旧继承方式。
- `dreamdojo_trocar_grpo_trial20.yaml`：仅 train 的 batch8 配置设 `wm_offload_diffusion_model=false`、`wm_offload_tokenizer=false`，同时保留外层 `enable_offload=true`。rollout 内常驻，actor 更新前卸载；独立 eval 的 `enable_offload=false` 使 DiT／VAE 常驻、文本缓存后留 CPU。
- DreamDojo 通用 pipeline 构造默认 cache／skip 仍是 false，不改其他调用者。所有开关均可单独回退；要完全回到旧行为，将 cache／skip 设 false、三个 split offload 设 null（包括 trial 的覆盖项）。
- 未改变去噪步数、图片分辨率、动作 chunk、reward 判定、full55 分片或每 5 轮评测协议。没有启用图像 latent 缓存、降低到 5 步、扩大到每卡 16／24 env、修改 FSDP 通信或重启任何设备。
- **20 轮 trial 尚未启动**。本次五卡第 2 轮实测 12.14 分钟；若相近，20 轮纯训练预算约 4 小时，再留初始化及外部 full55 评估尾部余量。这是两轮外推，不是完成时间承诺；优化后完整 full55 的两卡队列耗时尚未实测。
- 剩余性能主项是 WM 去噪和多卡 actor 更新。第二轮 rollout 8.17 分钟、actor 3.35 分钟；后者在新五卡配置里已不可忽略。增大 micro-batch、减少 FSDP 通信等需要另做同配置 profile 和显存测试，不能沿用旧单卡 smoke 的 35 秒更新估计。

## 复现入口

在 RLinf 根目录，使用 `.venv/bin/python`；沿用原实验的线程数 1、离线权重和 `NCCL_P2P_DISABLE=1`。执行前确认设备空闲，单卡诊断必须使用健康 GPU UUID；输出目录换成未存在的新目录。

```bash
# 单 chunk 同输入 A/B，batch8；不降低 35 步。
CUDA_VISIBLE_DEVICES=GPU-3b9e2a3e-1e46-2d99-9981-ed43b55d63a9 \
  .venv/bin/python toolkits/world_model/dreamdojo_perf.py micro \
  --steps 35 --repeats 2 --output logs/dreamdojo_repeat/perf_micro35

# 真正由 SFT 预测动作的 8 秒闭环；两组保持相同 batch／seed。
CUDA_VISIBLE_DEVICES=GPU-4a947c2b-b835-07b0-a588-bc2869411e5b \
  .venv/bin/python toolkits/world_model/dreamdojo_perf.py policy \
  --episodes 0 1 2 --cases baseline_resident resident --steps 35 --chunks 20 \
  --output logs/dreamdojo_repeat/perf_policy35

# 两轮优化后的真实训练共置 smoke，不是 20 轮 trial。
env -u CUDA_VISIBLE_DEVICES .venv/bin/python toolkits/world_model/dreamdojo_perf.py grpo \
  --gpus 0-3,5 --train-envs 40 --steps 35 --chunks 20 \
  --output logs/dreamdojo_repeat/perf_grpo35
```
