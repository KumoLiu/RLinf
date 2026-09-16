# 代码与文档收尾记录

日期：2026-09-16。范围：RLinf / DreamDojo / 相邻Isaac-GR00T。
本次目标是整理已完成工作，不重新训练、不改变reward/GRPO语义、不提交cluster作业。

## 1. 文档结构

RLinf `docs/dreamdojo/` 的27篇Markdown收敛为5篇：
README（入口）、RESULTS（结论）、RUNBOOK（运行）、IMPLEMENTATION（实现）、本文件（变更记录）。
保留 `cluster.env.example`，更新说明链接，不改环境变量值。

| 原材料 | 合并后的主位置 |
| --- | --- |
| FINAL/FULL/SWEEP_REVIEW、TRIAGE、FOLLOWUP、OVERNIGHT、WM_COMPARISON | RESULTS；完整逐实验数据仍在原logs CSV/JSON |
| BEST_CKPT_VIDEO_EVAL、BEST_BASE_VIDEO_COMPARISON、REWARD_MOTION_REVIEW | RESULTS + IMPLEMENTATION |
| CLUSTER、SLURM_CHAIN、BASELINE、GR00T_UPGRADE | RUNBOOK + IMPLEMENTATION |
| LAM_CHECKPOINT_AUDIT | IMPLEMENTATION严格加载与调用边界 |
| ACTOR_MICROBATCH_BENCHMARK、BATCH_SCALING、history/PERFORMANCE_REVIEW、OPTIMIZATION_LOG | RESULTS耗时 + IMPLEMENTATION性能契约 |
| KIR_HANDOVER、history/IMPLEMENTATION_REVIEW、FIX_LOG、HANDOFF、NATIVE_TRAIN_LOG | IMPLEMENTATION + RUNBOOK |
| 旧README/CLEANUP与根DREAMDOJO_HANDOFF | 重写为当前入口/整理记录，不保留过时“当前配置” |

DreamDojo保留upstream使用文档及EVAL_METRICS：
项目的EXPERIMENT_CONTEXT、HF_TELEOP_ROLLOUT_POSTTRAIN_RUN、SLURM_SWEEPS、
PICK_TROCAR_BACKUP合并为TROCAR_PROJECT；
MILESTONE_REWARD与MILESTONE_V3合并为MILESTONE_REWARD。
原始记录逐字保留在仓库外备份，不再在活跃docs中维护第二份结论。

## 2. 删除和保留的代码

已删除（均可从备份恢复）：

- RLinf：`toolkits/world_model/dreamdojo_actor_perf.py`、
  `dreamdojo_perf.py`：一次性actor/WM性能测量，结果已归纳，无生产调用方。
- RLinf：`tests/unit_tests/test_dreamdojo_actor_perf.py`：只测试已移除benchmark辅助函数。
- DreamDojo：`scripts/finish_lora_vs_full_eval.sh`、
  `resume_lora_vs_full_eval.sh`、`run_lora_vs_full_long_eval.sh`、
  `launch_cluster_lora_long.sh`：旧作业的等待/续评/传输包装；
  使用现有通用sweep与eval入口。
- DreamDojo：`tmp.slurm`：无调用方、属于另一项目的GR00T N1.5临时副本。

`test_dreamdojo_inference_perf.py` 改名为 `test_dreamdojo_inference.py`，
保留缓存、guidance快速路径、offload/VAE与报告读取回归；
只删除依赖旧benchmark的重复eval-return-None测试，该契约仍由diagnostics测试覆盖。

保留全部正式env/adapter/reward/KIR/video audit代码、native train/eval、container与续跑、
WM训练配置、通用数据/评测工具及有实际功能的测试。
`dreamdojo_validation.py` 是隔离问题的工具，不是“无用test”，因此保留。
v3分类器实验仍待人工复核，保留实现和测试，但文档明确未部署。

DreamDojo的通用LoRA提交器仅精简help中的重复实验分析，提交逻辑不变；
迁移工具的文档引用改为TROCAR_PROJECT；删除dataset里一行早已注释掉的print。
通用 `benchmark_latency.py` 支持指定权重/步数，仍有复用价值，因此保留。

## 3. 恢复与审计

修改前的恢复目录（本机，不在训练logs内）：

```text
/localhome/local-yunl/code_cleanup_archive/20260916_project_closeout.bA84wJ
```

备份196个文件，包含全部本次删除项、原项目docs、相关源码及修改前git状态/diff；
`backup_manifest.json` 记录逐文件SHA256。
原文位于 `<archive>/<repo>/files/<原相对路径>`。
恢复时先检查目标当前差异，再取需要的文件；不要整库覆盖或使用hard reset。

`cleanup_manifest.json` 记录本轮相对备份的新增/修改/移除及前后hash。
备份仅在本机；若将整个项目交接至另一机器，需要同时保留这份历史档案。
git原已提交文件也可从旧commit查看，但未提交的历史资料必须依赖本备份。

未动：数据、模型权重、训练logs、TensorBoard、视频/网页、PPT、环境与cluster runtime。
Isaac-GR00T原本干净，本次未修改。
DreamDojo原有checkpoint/dataset README及nightly docker文件的删除是既存改动，
不是这次清理，未恢复或扩大。其余既有dirty功能改动也未重置。
本次没有创建commit；后续提交时应逐文件审阅，避免把其他既有工作一并提交。

## 4. 验证记录

清理前：159项DreamDojo CPU回归通过。
清理后：151项RLinf回归 + 8项分类器v3回归全部通过。
少掉的8项为旧actor benchmark的7个参数化测试及1个重复诊断测试，
不是通过跳过失败测试获得通过。

其他检查：

- Ruff check / format检查通过（改名的inference测试）；两仓库git diff --check通过。
- 24个保留的shell/Slurm脚本通过bash -n；38个本地文档链接有效。
- 原生Hydra配置解析通过；最佳实验sweep dry-run通过，未提交作业。
- LoRA提交器4种recipe × train/eval共8组dry-run输出与备份逐字一致；
  删除help后的脚本文本相同，dataset删除注释前后Python AST相同。
- 196份备份SHA256全部核对通过；改动清单见cleanup_manifest.json。
- 项目文档33篇→7篇（RLinf 27→5，DreamDojo 6→2），正文行数减少约87%；
  不含upstream手册、根入口和保留的原始实验产物。

完整输出保存在上述恢复目录，不另建实验log目录。
测试有既有依赖弃用/Hydra provider警告，无失败。
原生配置dry-run退出0并输出完整可解析YAML，但MSC可选配置源因未设置profile记录了
错误堆栈和provider警告；不影响本次本地配置解析，未在清理时改依赖来屏蔽它。
本轮没有GPU重训/重评。
