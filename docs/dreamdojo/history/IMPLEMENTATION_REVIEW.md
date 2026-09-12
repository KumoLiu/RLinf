# DreamDojo × GR00T N1.7 × RLinf 实现审查

> 历史检查记录；当前入口和配置见 [README](../README.md)。

审查日期：2026-09-10。本文记录代码审查及已完成的诊断结果，不代表问题已经修复。

后续状态：本文保留修复前审查结论；本次实际修改、源码快照、测试与 GPU 对照结果持续记录在 [DREAMDOJO_FIX_LOG.md](DREAMDOJO_FIX_LOG.md)。V1 的 5／35 步对照现已完成：固定真实动作的重建指标未见 35 步优势，但策略闭环明显敏感，旧 step20 在三个起点上由 5 步的 [2,2,2] 变为 35 步的 [3,3,3]。应区分这两种实验，不能仅凭配置差异归因模糊，也不能凭 GT 回放否定采样步数对策略闭环的影响。

## 1. 结论

后续初始化决定：按用户要求，旧 step20 不再作为训练初始化或评估候选；后续从原始 SFT checkpoint 开始。本文及实验记录保留旧 step20 数据仅用于历史追溯，具体决定见 [修复记录](DREAMDOJO_FIX_LOG.md)。

当前接入存在四处已确认的训练／推理不一致：物理关节动作被裁剪、world model 动作归一化额外裁剪、遗漏训练时的状态条件，以及 GR00T 与 DreamDojo 的动作时间间隔不一致。应先修正这些接口问题，再判断 world model 的质量、数据量是否足够，以及 GRPO 能否提升策略。

另外，5 步去噪与原生评估 35 步的差异，以及用动作目标代替实际 proprioception 的近似，需要独立验证。目前没有证据能够量化每个问题对 success 或视频质量的贡献。

| 编号 | 问题 | 状态 | 优先级 |
| --- | --- | --- | --- |
| F1 | 解码后的物理关节目标被加噪并裁剪到 `[-1,1]` | 代码与真实数据已确认 | P1：恢复训练前处理 |
| F2 | DreamDojo 在线 min-max 归一化多了一次裁剪 | 已用真实动作窗口量化 | P1：恢复训练前处理 |
| F3 | 在线 384 维条件遗漏训练时写入的状态 | 已执行原生变换函数复现 | P1：恢复训练前处理 |
| F4 | 30 Hz 策略动作被重复后按 15 Hz 执行 | 实际 checkpoint 与代码已确认 | P1：明确时间轴后处理 |
| V1 | 使用 5 步去噪，原生评估默认 35 步 | 配置差异已确认，质量影响待对照 | P2：质量基线验证 |
| V2 | 下一步 proprio 直接使用最后一个动作目标 | 近似已确认，闭环影响待验证 | 结构性限制 |

P1 表示会改变当前模型实际输入或执行语义，不表示已通过修复后的成功率实验证明其影响大小。

## 2. 审查范围与实验背景

审查对象：

- `/localhome/local-yunl/RLinf`：DreamDojo 环境、动作适配器、奖励适配器、GR00T RL 路径、配置、测试和实验日志。
- `/localhome/local-yunl/DreamDojo`：原生数据变换、G1 数据转换、推理入口、LoRA 配置和 milestone reward 实现。
- `/localhome/local-yunl/Isaac-GR00T`：G1-Dex3 模态配置；同时读取实际使用 checkpoint 的 processor 配置，避免只依赖仓库示例。

RLinf 审查基线为 `36996329` 加当时的未提交改动。该提交包含 DreamDojo 初始接入，审查也覆盖其相对上一个上游提交 `1c9eed00` 的新增实现。结论针对审查时的本地文件，不能假定后续改动仍保留同样行为。

参考方案为 [RLinf 官方 WAN world-model RL 示例](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)。该示例使用生成视频作为观测，并以奖励模型提供 RL 信号；迁移时仍需分别对齐新模型的动作、状态、时间采样和推理配置。

实际实验入口：

- [训练配置](examples/embodiment/config/dreamdojo_trocar_grpo_gr00t_n1d7.yaml)
- [环境配置](examples/embodiment/config/env/dreamdojo_trocar.yaml)
- [启动脚本](examples/embodiment/run_dreamdojo_trocar.sh)

实际模型：

```text
GR00T:
/localhome/local-yunl/models/s2r_models_dev/yunl/sft/g1/pick_trocar/
  g1_pick_trocar_head_10k_bs32_lr1e-4

DreamDojo:
/localhome/local-yunl/models/dreamdojo/lora_r32_lr3e-4_r64_18k/
  checkpoints/iter_000018000/model_ema_bf16.pt

Reward:
/localhome/local-yunl/DreamDojo/outputs/milestone/v2/best.pt
```

训练配置为 GRPO，group size 8，192 个训练环境；评估配置为 56 个环境。训练及评估均使用 DreamDojo，当前日志中的 eval 不是实机成功率。每次生成 12 个新视频帧，条件图像 1 帧，分辨率 480×640，episode 上限 120 步。

读取本机数据 metadata 得到的规模如下。这里统计的是轨迹数，不代表样本独立性或有效覆盖度。

| 数据源 | train | validation | 标称帧率 |
| --- | ---: | ---: | ---: |
| teleop success | 225 | 25 | 30 fps |
| rollouts 30k bs256 | 151 | 17 | 30 fps |
| rollouts 10k bs32 | 115 | 13 | 30 fps |
| 合计 | 491 | 55 | — |

## 3. F1：物理动作被错误裁剪

位置：[gr00t_action_model.py](rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py)，`predict_action_batch` 和 `_apply_exploration_noise`，审查时约第 985、1083 行。

调用顺序为：

```text
策略输出归一化动作
→ 反归一化，臂动作从相对表示恢复成绝对关节目标
→ 拼接为 28 维 G1-Dex3 动作
→ 加 Gaussian noise
→ clamp(-1, 1)
→ 送给 DreamDojo
```

关键代码：

```python
noise_scale = float(self.action_head.rl_config.get("action_noise_scale", 0.1))
noise = torch.randn_like(raw_tensor) * noise_scale
raw_tensor = (raw_tensor + noise).clamp(-1.0, 1.0)
```

当前 DreamDojo recipe 继承 [GR00T 默认配置](examples/embodiment/config/model/gr00t_n1d7.yaml) 的 `action_noise_scale: 0.1`，因此这段代码实际启用。它位于通用 GR00T RL 实现中，问题是接入 G1 时继续使用了不适合其物理动作范围的默认处理。

已验证的真实数据范围：

- 225 条 teleop train 中，左臂目标范围约 `[-1.158, 1.955]`。
- 左手目标范围约 `[-1.047, 0.960]`，右手约 `[-0.960, 1.047]`。
- 手部每组约 28.57% 的动作元素本来就在 `[-1,1]` 之外。
- 将 teleop train episode 0 的前 12 个真实动作送入现有后处理函数，固定随机种子为 0，约 9.52% 的输出元素落在裁剪边界，平均绝对改变量约 `0.0721`。

影响：合法关节目标会被截断，并叠加以物理关节单位计的噪声。该处理仅在 `mode="train"` 启用，eval 跳过，造成额外的执行分布差异。

建议：先以 `action_noise_scale: 0.0` 建立无此后处理的基线，再单独验证探索设置。若需要物理动作限幅，应按真实关节限制和动作单位定义；不要把归一化动作边界直接用在物理目标上。GR00T 的 flow-SDE 路径本身已有探索机制。

## 4. F2：world model 动作归一化与训练不一致

位置：

- 在线：[dreamdojo_adapters.py](rlinf/envs/world_model/dreamdojo_adapters.py)，`G1DreamDojoActionBridge._normalise`，约第 129 行。
- 训练：[DreamDojo state_action.py](../DreamDojo/groot_dreams/data/transform/state_action.py)，`Normalizer.forward` 的 `min_max` 分支，约第 146 行。

两边都使用官方 `shared_meta/G1_stats.json` 的 min/max 做变换，但只有在线实现额外执行：

```python
return normalised.clamp(-1, 1)
```

原生训练的 min-max 分支不执行这个裁剪。这里不能假设归一化结果天然落在 `[-1,1]`：统计量来自官方 G1 数据，并非当前任务每个关节目标的严格边界。

验证方法：遍历 225 条 teleop train 的 parquet；每 12 个原始时间步取一个完整的 25 行动作窗口，按原生规则隔帧取样，再比较三个四动作块的相对差值。两边使用相同原始动作、维度映射和统计量，只比较额外裁剪造成的差异。

结果：

| 指标 | 结果 |
| --- | ---: |
| 完整动作窗口数 | 2,486 |
| 条件张量出现差异的窗口数，判定阈值 `1e-6` | 1,801 |
| 最大单元素绝对差异 | 1.279908 |
| 最大差异样本 | `episode_000000.parquet`，窗口起点 156 |

episode 0 的原生归一化动作范围约为 `[-2.995, 2.280]`。部分初始静止窗口的块差值仍然相同，所以只测 episode 开头或合成的小范围数据会漏掉问题。

影响：先裁剪再相减会改变块相对动作，可能压缩或抹掉真实关节运动。即使输入完全来自 GT 数据，在线条件也可能与 world model 训练时不同。

建议：对现有 checkpoint 精确复现训练变换，移除这次额外裁剪。不要只在推理端改用另一份统计量；更换统计量也会改变 checkpoint 所对应的输入语义。

## 5. F3：遗漏训练时实际写入的状态条件

位置：

- 训练：[DreamDojo dataset.py](../DreamDojo/groot_dreams/data/dataset.py)，`WrappedLeRobotSingleDataset.__getitem__`，约第 1093 行。
- 在线：[dreamdojo_adapters.py](rlinf/envs/world_model/dreamdojo_adapters.py)，`encode_30hz_actions`，约第 166 行。

原生代码先把 G1 动作填到 `58:101`，之后又执行：

```python
key = action_seq[0:1, :29]
key[:, :min(original_outputs["state"].shape[1], 29)] = original_outputs["state"][:, :29]
```

`key` 是张量视图，赋值会修改 `action_seq` 本体。因此，训练样本除了 G1 动作，还在第一行的前 29 维包含归一化状态。按 43 维 G1 布局，这一段包括腿、腰、左臂和左手；不能直接以原始 28 维 Dex3 状态代替。

在线实现创建全零 384 维张量，只填 G1 动作槽：

```python
encoded[..., DREAMDOJO_G1_SLICE] = deltas
```

它没有接收或填入上述状态条件。网络的 action embedder 消费完整 action 张量，所以这不是仅影响日志或 `__key__` 的差别。

验证：使用 teleop train episode 0 的真实 state/action 和训练统计量，执行现有原生 `__getitem__` 函数中的变换逻辑。验证使用预先构造的变换输入，并绕过 CUDA-only metadata 创建，不涉及 world model 推理。结果为：

- 原生 action shape：`[12,384]`。
- 原生第一行前 29 维非零数：29；在线：0。
- 缺失状态条件的最大绝对值：约 `2.59293`。
- 原生第一行前 29 维与按训练规则归一化的状态完全相同。

建议：明确这段状态写入是否符合长期设计，但使用当前 checkpoint 时，应先匹配其实际训练输入。若要去掉这一隐式写入，需同步修改数据管线并重新验证或训练模型，不能只在推理端省略。

## 6. F4：策略动作与 world model 的时间尺度不一致

位置：

- [实际 GR00T checkpoint processor](../models/s2r_models_dev/yunl/sft/g1/pick_trocar/g1_pick_trocar_head_10k_bs32_lr1e-4/processor_config.json)
- [G1-Dex3 模态配置](../Isaac-GR00T/examples/g1-dex3/g1_dex3_head_config.py)
- [DreamDojo G1 时间采样](../DreamDojo/groot_dreams/groot_configs.py)
- [在线动作重复](rlinf/envs/world_model/dreamdojo_adapters.py)，`encode_policy_chunk`，约第 184 行。

实际 checkpoint 的 `new_embodiment` 动作 `delta_indices` 是 `[0,1,...,15]`，源数据为 30 fps。DreamDojo 的 G1 数据变换则使用 `timestep_interval=2`，即按 15 Hz 视频时间轴建模。

在线实现取 12 个策略动作，每个重复两次，再按隔帧规则采样：

```python
future_30hz = torch.repeat_interleave(actions_28, 2, dim=1)
raw_window = torch.cat([previous_action_28[:, None], future_30hz], dim=1)
return self.encode_30hz_actions(raw_window)
```

这实际上把连续 30 Hz 的策略目标作为连续 15 Hz 目标使用。12 个命令原本对应约 0.4 秒的控制区间，现在被铺到约 0.8 秒。它改变了执行速度，并不等同于从真实 30 Hz 动作序列正确下采样。

此外，还需要明确图像与动作 baseline 的相位。原生 [dataloader](../DreamDojo/groot_dreams/dataloader.py) 多取一个抽样帧作为基准；[dataset.py](../DreamDojo/groot_dreams/data/dataset.py) 随后丢弃它。例如原始采样索引为 `0,2,...,26` 时，模型条件图像是 `I2`，首个动作差值是 `a2-a0`。在线 reset 使用 `I0` 和数据里的 `a0` 作起点。必须按当前图像时间、历史基准动作时间和预测动作时间统一检查，不能只对齐张量长度。

影响：物体与手部运动的时间尺度改变，策略重新规划时收到的观测也不同于 SFT 的时间语义。`max_episode_steps=120` 同时被用作 120 个策略命令和 120 个 WM 新帧，掩盖了两种时间尺度的差别。

建议：先明确统一的真实时间轴，分别定义策略执行命令数、WM 条件动作数、WM 消费帧数和 episode 秒数。保留 30 Hz 策略时，需要设计一致的重采样和重新规划方式；若改为 15 Hz 策略，应重新验证 SFT 动作时间配置。不要仅通过重复命令满足 WM 的固定输入长度。

## 7. V1：5 步去噪的质量尚未验证

当前 [环境配置](examples/embodiment/config/env/dreamdojo_trocar.yaml) 设置 `num_inference_steps: 5`，实际调用也传入该值。

DreamDojo [原生推理入口](../DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py) 的默认 `num_steps` 为 35；[原生验证脚本](../DreamDojo/scripts/validate_pick_trocar_world_model.py) 和 [批量验证脚本](../DreamDojo/scripts/run_hf_posttrain_val_eval.sh) 也默认使用 35 步。

WAN 示例的 5 步配置适用于其配套模型，不能直接证明当前 DreamDojo LoRA checkpoint 的低步数生成质量。已检查的配置未显示针对 5 步推理的蒸馏或质量对照。

这可能参与造成模糊、细小 trocar 结构退化和闭环误差积累，但本次未执行同输入的 5／35 步生成对照，因此不把它定为已证实的模糊根因。

建议在修正条件编码后，固定 checkpoint、真实初始帧、GT actions 和 seed，比较 5 与 35 步，并分别检查单个 chunk、teacher forcing 和多 chunk 闭环。先建立质量基线，再减少推理步数。

## 8. V2：proprioception 是动作目标的代理值

位置：[world_model_dreamdojo_env.py](rlinf/envs/world_model/world_model_dreamdojo_env.py)，约第 564 行；[dreamdojo_adapters.py](rlinf/envs/world_model/dreamdojo_adapters.py)，`update_g1_dex3_state`。

当前每个 chunk 后执行：

```python
self.current_state = update_g1_dex3_state(self.current_state, actions_tensor[:, -1])
```

该函数直接返回动作目标的副本，即假设关节准确到达最后一个命令位置。DreamDojo 并未预测这个实际状态。

GR00T checkpoint 使用 proprio，而且臂动作的相对表示需要根据当前 state 恢复为绝对目标。若生成图像显示的手臂位置与这个代理 state 不一致，就会把不匹配的视觉和状态输入下一步策略。

真实 teleop train 的同帧 action 与 observation.state 平均绝对差约为：左臂 `0.0203`、右臂 `0.0192`、左手 `0.0458`、右手 `0.0267`。这些数值说明两者并不相同，但它们不是该在线代理状态误差的直接测量，也不能单凭这些数值判定近似不可用。

建议用真实轨迹验证代理 state 对动作解码和闭环行为的影响，并显式记录这一限制。若不能满足闭环需求，再评估状态预测模型或适配缺少 proprio 的策略方案。

## 9. 已完成的其他验证

### 9.1 Reward 适配器可识别真实成功视频

使用当前 `BatchedMilestoneReward` 和 `v2/best.pt`，在 GPU 上读取 teleop validation 的 episode 0、1、2。将原始 30 fps 视频隔帧采样为 15 fps，保留 `duplicate_for_30fps=True`，模拟 WM 输出频率。

| Validation episode | 原始帧数 | 最终 stage | 总 reward | 三个里程碑首次确认的原始帧索引 |
| --- | ---: | ---: | ---: | --- |
| 0 | 201 | 3 | 3 | 74、180、190 |
| 1 | 209 | 3 | 3 | 120、190、200 |
| 2 | 177 | 3 | 3 | 82、150、168 |

这排除了奖励适配器对所有输入都无法前进的情况，但样本只有三条，且都是成功的真实视频。尚未验证生成视频上的召回率、失败轨迹上的误报率或 reward hacking 风险；不能据此宣布奖励模型已充分校准。

### 9.2 实际日志中 eval 连第一个里程碑都未确认

主要实验目录：

```text
logs/20260910-01:41:58-dreamdojo_trocar_grpo_gr00t_n1d7/
```

检查了 `metrics.log`、`run_embodiment.log` 以及 TensorBoard events 中的原始 scalar：

- 已记录的 22 次训练统计中，`env/success_once` 约为 `0～0.0104167`。
- `env/return` 约为 `0.0104167～0.109375`，存在少量 milestone 奖励，不能说训练完全没有梯度信号。
- TensorBoard step 4、9、14、19 的四次评估中，`eval/success_once` 和 `eval/return` 均为 0。
- `logs/step20_full_eval/` 的额外评估同样记录 success 和 return 为 0。

当前奖励只在确认新里程碑时支付正奖励，因此 eval return 为 0 表示没有确认任何里程碑，不只是没有完成最终放置。

这些日志没有逐轨迹保存三头概率及阶段轨迹，无法仅靠汇总指标区分“没有抓起”和“视频质量导致奖励漏检”。旧实验也未保存完整的未提交源码快照，因此不能把当前工作区中每行代码的行为都当作旧实验已被逐行复现。

### 9.3 未发现明显的 DreamDojo checkpoint 缺键或形状错误

主要训练日志记录了 rank 64、alpha 64 的 LoRA 配置，加载信息为：

```text
missing_keys=[]
unexpected_keys=[]
incorrect_shapes=[]
```

这支持所选权重与模型结构在键和形状层面兼容，不能证明 checkpoint 本身质量足够，也不能排除所有推理语义问题。

28 维 Dex3 到 43 维 G1 的四组关节重排与离线转换脚本一致。输入图像与生成视频的 `[-1,1]`／uint8 转换也未发现明确的整体范围错误。

### 9.4 现有测试全部通过，但没有覆盖关键一致性

执行命令，工作目录为 RLinf 根目录：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit_tests/test_dreamdojo_adapters.py
```

结果：`9 passed in 6.54s`。

测试缺口：

- 归一化测试使用范围内的合成值，没有覆盖真实数据超出官方统计范围的情况。
- 测试断言 G1 动作槽以外为零，反而把遗漏状态条件的行为当作正确预期。
- 没有从真实 checkpoint 的动作时间索引检查 30 Hz／15 Hz 转换。
- `update_state` 测试只证明函数返回动作目标，没有验证代理状态的物理或视觉一致性。
- 没有覆盖 GR00T 解码后对物理关节目标的加噪和裁剪。

现有测试通过不能作为训练／推理数据已经对齐的依据。后续应增加真实轨迹的输入一致性检查，预期值来自原生数据流程，而不是再次复述在线实现。

## 10. 建议修复和验收顺序

1. **对齐数据契约。** 修复 F1～F4，明确关节顺序、单位、相对／绝对语义、统计量、状态条件槽、图像时间、历史动作基准和输出时间轴。用真实轨迹窗口比较原生训练输入与在线编码，覆盖静止、抓取、交接、放置及超出官方统计范围的样本。
2. **建立 GT 动作下的 WM 基线。** 固定初始观测和动作，以 35 步检查单 chunk、teacher forcing 和闭环视频。再用相同输入比较 5 步，区分接口问题、单步模型误差和闭环累积误差。
3. **验证生成视频上的奖励。** 对真实成功／失败和对应生成视频保存三头概率、stage、每帧 payout，并人工检查小物体抓取和放置，量化误报与漏报。
4. **建立未做 RL 的 GR00T 闭环基线。** 使用固定评估初始状态、明确动作执行频率和 episode 秒数，记录动作范围、state 代理值及视频。先确认 SFT 策略在当前 WM 中能得到有意义的里程碑反馈。
5. **恢复小规模 GRPO。** 记录每组 reward 是否存在差异、全零组比例、每阶段完成率和训练／评估差异，再逐步扩大并行环境数。

修复后至少应满足：真实轨迹条件编码差异可解释且在预定数值容差内；合法物理动作不再受归一化区间错误限幅；策略与 WM 时间轴一致；SFT 基线、GT 回放质量和奖励识别能力都有独立记录。不要仅以“训练能够运行”作为验收标准。

## 11. 本次审查的边界

- 已完成源码审查、真实数据 CPU 数值对照、三条真实视频的奖励推理、现有单元测试和日志检查。
- 数值对照为审查时的诊断运行，未新增持久化诊断脚本；本文保留了数据源、采样规则及测量结果。
- 未执行 DreamDojo 5／35 步生成对照，未重跑完整 WM 或 GRPO 训练，未做实机评估。
- 没有修改模型、环境、训练配置、奖励阈值或 checkpoint；本次交付仅新增本文档。
- TODO(agent)：修复后补充真实数据一致性测试和上述对照结果，再更新每项发现的解决状态。
