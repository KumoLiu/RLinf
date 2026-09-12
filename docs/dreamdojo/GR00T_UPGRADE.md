# GR00T N1.7 本地源码升级与评测（2026-09-11）

## 实际升级

- RLinf editable 源码仍位于 `.venv/gr00t`。
- 从 `23ace64f17aa5015259b8609d371eb61a357c776` 切到
  `2d9a9fce9811d10510d1263344fbf5253f33639d`，detached HEAD 固定版本。
- Git 对象来自本地 sibling `Isaac-GR00T`，没有修改该 sibling 仓库。
- 旧版保留在 `.venv/gr00t-23ace64-baseline` worktree。
- 未升级 Python/PyTorch，未重新安装 pip 包，未更换 SFT 权重。
  当前 Python 3.10.21 / PyTorch 2.7.0+cu128 与新版 pyproject 声明的
  Python 3.12 / PyTorch 2.9 不同。本次验证的是新版源码在当前 RLinf 组合环境中的兼容性，
  不是新版官方完整依赖栈；editable 安装元数据仍来自旧安装，源码版本应以 Git 为准。

需要回退时，先停止使用该源码的进程，再从 RLinf 目录执行：

```bash
git -C .venv/gr00t switch --detach 23ace64f17aa5015259b8609d371eb61a357c776
```

## 升级前隔离检查

44 项 CPU 回归通过；同一 SFT checkpoint、3 个验证初始状态，新版 train/eval
动作生成和训练 log-prob 回放均为有限值，未执行反向传播或参数更新。

已确认 checkpoint 配置 `letter_box_transform=false`：旧版仍补黑边，新版遵循此配置。
真实 480×640 图像在 VLM 前的预处理输出由 256×256 变为 256×340；该测试帧旧版
约 20.7% 黑像素，新版无黑像素。3 个初始状态的 VLM image grid 从 `[1,16,16]`
变成 `[1,16,22]`。动作存在变化，但这不证明成功率提升。

RLinf 训练 Flow-SDE 和 log-prob 实现未改；eval 使用上游去噪循环，该循环在两个
commit 间未变。上游 processor/backbone 仍参与模型推理，因此输出可能变化。

## 本次完整评测

- 使用仓库已有 `evaluations/eval_embodied_agent.py` / `EmbodiedEvalRunner`。
- 主 YAML 为 `examples/embodiment/config/dreamdojo_trocar_grpo_gr00t_n1d7.yaml`。
- CLI 指定 `runner.only_eval=true` 和 `'rollout.model=${actor.model}'`，使 standalone
  rollout 使用完整的 SFT 模型配置，而非训练配置中仅有路径和精度的 rollout 子配置。
- 没有改评测源码、启动包装器、种子逻辑、奖励算法或去噪算法；没有创建 actor 训练任务。
- 7 个 worker 使用物理 GPU 0–3、5–7；56 个槽位覆盖 55 个唯一案例，padding 不计分。
- 同一 SFT base、policy 4 步、WM 35 步、240 条命令、原 milestone v2 reward，保存视频。
- 未添加 policy 固定种子，保持原生采样行为。与历史结果的差异包含随机波动，不能
  称为严格配对或统计显著的版本差异；历史外部脚本的 SFT 15/55 仅供参考。

运行目录：`logs/20260911-gr00t-new-native-eval/`。
启动时间：2026-09-11 06:13:53 UTC；driver PID：1677113。
`console.log` 首部记录了实际源码路径、commit 与完整原生入口命令。
只保留原生日志、配置、指标和视频，不复制 source 或 `.diff`。

## 完成结果

2026-09-11 06:28:02 UTC 输出完整指标，进程随后结束，GPU 显存全部释放。
原生 eval 阶段耗时 573.808 秒（9 分 34 秒）；从启动到指标输出约 14 分 9 秒，
其中包含 Ray/worker 启动及模型加载。

| 指标 | 本次结果 |
| --- | --- |
| num_trajectories | 55，padding 未计入 |
| picked | 39/55 = 70.91% |
| handed | 26/55 = 47.27% |
| placed / success_once | 25/55 = 45.45% |
| 平均 return | 1.63636 |
| picked_prob_max（跨案例平均） | 0.55714 |
| handed_prob_max（跨案例平均） | 0.47156 |
| placed_prob_max（跨案例平均） | 0.46647 |

这里的 success 是 WM 生成视频上的 milestone classifier 判定，不是真实机器人成功率。
相对历史外部脚本 SFT 15/55（27.27%），本次高 10 个案例、18.18 个百分点；
由于评测入口、分片/随机数消费不同，不能把差异全部归因于升级。
本次没有重新跑旧版相同原生入口，也不要用 noise=0.1 的 RL step5（10/55）
作为源码 A/B 对照，因为其权重已经训练过。

本地升级后重新运行 44 项现有 CPU 回归，全部通过（18.08 秒，无跳过）；
记录为运行目录的 `regression.xml`。未增加固定种子或修改原生 eval 代码。

视频位于 `video/eval/seed_0/0.mp4` 至 `seed_6/0.mp4`，每个为 8 个环境的 2×4 拼图。
全部完整解码成功：2560×960、15 fps、121 帧、8.0667 秒。最后一个 worker
含 1 个只参与 batch 计算的 padding 画面，并不多计成功率。

抽查 `seed_0`：case 0 的手臂和器械有明显运动，生成后段主体轮廓仍可辨，未见整段
画面崩成模糊色块；局部手部/阴影仍有生成伪影。该抽查不能代表全部 55 个案例，
也不是新旧 WM 清晰度的定量比较。预览为 `episode_0_contact.jpg`（0–7 秒每秒一帧）
与 `seed_0_final.jpg`（约 7.9 秒的 8-env 拼图）。

下一道归因检查是旧版相同原生入口的 SFT eval，必要时重复多个随机运行；
本次未启动 RL 训练，也未验证新版源码下的反向传播。
