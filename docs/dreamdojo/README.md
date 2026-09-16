# DreamDojo × GR00T N1.7：项目入口

整理日期：2026-09-16。此目录记录已完成的神经模拟器 RL 工作，不再按每次排查/每次提交新增一篇文档。

## 结论先看这里

真机初步结果（用户报告）：任务 SFT base **44%**，RL **66%**，提升 **22 个百分点**。
测试的是 `wm_scratch_r32_n03_gb128_wm15_s1234` 的 **220 代，无 KIR**。
试验次数和逐次判定表尚未归档，因此不声称统计显著性。

RL 只更新 GR00T 的可训练参数；DreamDojo 与三阶段 reward classifier 冻结。
“scratch WM”是从官方预训练权重重新做任务适配，不是随机初始化世界模型。
保留的 SFT base 用于对比评测，不作为本轮的 KL reference。

## 文档导航

| 想知道什么 | 唯一主文档 |
| --- | --- |
| 实验结论、候选 checkpoint、真机与 WM 指标、训练耗时 | [RESULTS.md](RESULTS.md) |
| 环境、资产路径、原生 train/eval、Slurm 续跑、诊断与测试 | [RUNBOOK.md](RUNBOOK.md) |
| 动作桥、奖励、KIR、修复与仍存在的限制 | [IMPLEMENTATION.md](IMPLEMENTATION.md) |
| 本轮删改清单、历史文档去向、恢复办法、验证结果 | [CLEANUP.md](CLEANUP.md) |
| Cluster 环境变量 | [cluster.env.example](cluster.env.example) |

相关仓库的主记录（当前 workspace 中相邻 checkout）：

- [DreamDojo：数据、WM 训练与对比结论](../../../DreamDojo/docs/TROCAR_PROJECT.md)。
- [DreamDojo：v2 reward 与未部署的 v3 候选](../../../DreamDojo/docs/MILESTONE_REWARD.md)。

## 代码入口

- 配方：[dreamdojo_trocar_grpo_gr00t_n1d7.yaml](../../examples/embodiment/config/dreamdojo_trocar_grpo_gr00t_n1d7.yaml)。
- 本地原生训练：[run_dreamdojo_trocar.sh](../../examples/embodiment/run_dreamdojo_trocar.sh)。
- Cluster / 对比实验：[docker/dreamdojo/](../../docker/dreamdojo/)。
- 环境与 reward：[rlinf/envs/world_model/](../../rlinf/envs/world_model/)。
- 保留的定位工具：[dreamdojo_validation.py](../../toolkits/world_model/dreamdojo_validation.py)。

注意：YAML 的本地默认值不是最优实验配置。最优实验采用明确的 cluster overrides，
包括 scratch WM、noise 0.3、micro batch 8、global batch 128、WM train 15 / eval 35。
请按 RUNBOOK 复现，不要把旧的 7 卡 / noise 0.1 试跑当成最终方案。
