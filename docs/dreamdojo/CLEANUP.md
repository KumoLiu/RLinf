# 2026-09-11 代码整理记录

范围：准备 DreamDojo × GR00T N1.7 的 RL 对比实验与 cluster 迁移。
没有停止/重启训练、改运行中 trial 的配置、上传数据、提交 Git commit、push 或提交 Slurm 作业。

## 审查基线

| 仓库 | 当前 HEAD | 审查结果 |
| --- | --- | --- |
| RLinf | `36996329487374af2b42f87d87d88cda6d0eb3c9` | 11 个 tracked 修改，22 个 untracked 文件；本次重点精简 |
| DreamDojo | `783004974e67f8eeb7792a948b2acfc453833c5f` | 既有 WM 训练/标注/Slurm 改动与 RL 推理补丁混在一起，分开处理 |
| Isaac-GR00T | `2d9a9fce9811d10510d1263344fbf5253f33639d` | 工作树无本地修改 |

这些是基线 SHA，不包括尚未提交的必要修改。仅 push 原有 HEAD **不会**带上当前 pipeline。
本次不把“代码已整理”描述为 `git status` 全空；保留改动供审核后分主题提交。

## 删除、迁移与保留

1. 从 RLinf 移出 6 个已弃用文件：外部 `dreamdojo_trial.py`、`dreamdojo_full_eval.py`、
   两份对应测试、`dreamdojo_trocar_grpo_trial20.yaml`、`dreamdojo_trocar_grpo_long.yaml`。
   可恢复归档位于 `/localhome/local-yunl/code_cleanup_archive/20260911_dreamdojo/rlinf/`，
   保留原目录结构；这是历史材料，不能直接作为独立可运行工具。原先测试中的有效
   checkpoint 命名、组内奖励指标测试已迁移到 `test_dreamdojo_diagnostics.py`。
2. 移除 validation/perf 内重复的训练启动逻辑；训练与完整 eval 统一使用 native runner。
   删除自动源码快照、全仓库 diff、snapshot 比较输出；没有删除任何旧日志、视频或 checkpoint。
3. 将 6 份历史设计/检查/实验文档集中到 `docs/dreamdojo/history/`；根目录交接文档仅保留索引。
   历史文件保留当时的路径、数值和旧命令，当前使用方式以 README 为准。
4. 配方中的本机绝对路径改为环境变量覆盖，rollout/actor 共用 policy 路径。
   允许 site env 与集群解释器，不再强制使用本机 `.venv`。GPU 故障排除仍作为本机默认，
   不再在通用性能工具中硬编码本机的 GPU UUID/可用卡集合。
5. 删除无人调用的 `dreamdojo_action_deltas` / `encode_dreamdojo_actions`：旧版缺少
   normalization/state prefix，容易与正式 bridge 混淆。测试直接覆盖正式
   `G1DreamDojoActionBridge` 的 block delta、384D 布局和非法输入。
6. 删除 DreamDojoEnv 中已被初始化检查禁止的 KIR/auto-reset 分支和无消费方的
   `condition_action` / `retain_action` 缓冲。不新增 auto-reset 支持，不改变当前 rollout。
7. 删除 DreamDojo `build_net()` 临时 `FINAL MODEL CONFIG` warning；其余用户 WM 训练行为不改。

必须保留的 RLinf 修改：

- env/action/GR00T converter 注册；absolute target state 更新与取消物理动作 [-1,1] 截断。
- LeRobot frame2 初始化、frame0 的 action/state 历史、三数据源采样。
- 正确 28D→43D→384D、min/max、2×采样、4步 block delta、state_t prefix。
- batched milestone reward、30/15 Hz 对齐、视频完整帧记录、55-case padding mask。
- WM cache/residency 开关及 CPU/GPU 边界迁移；只在此配方关闭 FSDP sync_module_states。
- opt-in GRPO 组内 flat/zero reward 统计。其他模型的默认值不改。

必须随 RL 迁移的 DreamDojo 本地补丁：

- `cosmos_predict2/_src/imaginaire/lazy_config/lazy.py`：重复注册 resolver 兼容。
- `cosmos_predict2/_src/predict2/inference/video2world.py`：真实 batch 推理、prompt/action 对齐、文本缓存/offload。
- 新增 `cosmos_predict2/_src/predict2/inference/text_embedding_cache.py`。
- `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py`：
  opt-in guidance=0 跳过无条件支路。
- `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`：两处 sampler 的 batch 维修复。
- `groot_dreams/data/transform/state_action.py`：仅实际使用旋转变换才要求 pytorch3d。
- 当前 r64 实验 YAML 与原有 G1 statistics；README 列出确切文件。

DreamDojo 中暂不作为 RL 清理对象的既有修改：

- motion-consistency axis、WM LoRA/full/motion-loss 实验配置、sweep/submitted override；
- milestone 标注、训练、审核工具和模型输出；
- WM holdout/视频 metrics、可视化/比较脚本，材料化数据集索引；
- Slurm 的 LAM mount/preflight、数据同步脚本、ignore 规则；
- 既有 `checkpoints/README.md`、`datasets/README.md` 和三个 nightly Docker 文件的删除。

这些是用户既有工作，不因为当前 RL 不直接调用就删除或恢复。迁移时应单独提交/挑选，
不能简单 `git add -A` 把所有临时可视化、sweep 状态与 RL 部署混成一个提交。

## 本机已验证的软件版本

Python 3.10.21；torch 2.7.0+cu128；torchvision 0.22.0；transformers 4.57.6；
ray 2.58.0；hydra-core 1.4.0.dev1；omegaconf 2.4.0.dev4；gr00t 0.1.0；
cosmos-predict2 1.4.1；decord 0.6.0；torchcodec 0.5；opencv-python 5.0.0.93；
tensorboard 2.21.0。这是环境记录，不是完整依赖 lockfile；Torch/CUDA/GR00T 相关编译包不能只按名字随意升级。

## 验证与边界

最终 **44 项 CPU 回归全部通过**（10.63 秒，未跳过），覆盖正式 bridge、absolute state、
reset 历史字段、reward、视频/推理优化、7/8 worker padding 计数、cluster 路径覆盖和干净诊断输出。
原生入口 `--cfg job --resolve` 成功退出；shell 语法、相关 RLinf Python 文件 Ruff 检查/格式检查通过，
两仓库 `git diff --check` 通过。
Hydra 的 MSC profile 未配置 warning 属于已有环境噪声，不影响本地文件配置解析。

对所有变更的现存 Python/YAML 文件做语法检查：RLinf 18 Python + 2 YAML，
DreamDojo 38 Python + 21 YAML，全部通过。语法检查不等同于执行了用户既有的 WM/Slurm 工具。

真实验证集动作预处理 parity：

| 数据集 | 25-sample 窗口数 | 失败 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| teleop validation | 277 | 0 | 0.0 |
| rollout 30k val | 146 | 0 | 0.0 |
| rollout 10k val | 145 | 0 | 0.0 |
| 合计 | 568 | 0 | 0.0 |

parity 使用真实 parquet actions/state，对照 DreamDojo 原生 Normalizer 与 dataset transform 源码；
图像插值和 CUDA 调用被 stub，不是像素生成/完整 GPU dataloader 验证。
另外验证 reset 使用图像/state frame2、历史 action/state frame0。
本次输出只在 `/tmp/dreamdojo-cleanup.jiNDMX/`，没有向训练日志新增源码或 diff。

逐字段核对本次 YAML 与实际 trial 的 resolved config：policy、优化器、算法、路径、种子、
train/eval 和设备配置保持不变；输出目录不同。原生校验额外补入的
`grad_scaler.init_scale/growth_interval`、`sampling_params.max_new_tokens` 为 null，
不属于本次算法变更。trial 的 5 轮上限来自原有 CLI，主配置仍为 1000。

trial 在整理过程中按原计划自然结束，step5 eval=10/55（18.18%），
checkpoint 与 7 个 eval 视频已落盘，详情见 [实验记录](history/NATIVE_TRAIN_LOG.md)。未自动续训。

没有占用训练 GPU 重跑整条 pipeline，也没有验证目标 cluster 的容器、挂载和 NCCL 拓扑。
建议按 README 顺序先解析配置、跑 CPU 回归，再在 cluster 做短 GPU smoke 后启动成组实验。
