# DreamDojo 原生 train / eval 长训练

> 实验历史；当前入口和配置见 [README](../README.md)。

## 2026-09-11：noise=0.1 trial 完成

- 5/5 轮及原生 eval 完成，driver PID 1477962 已自然退出；代码整理期间没有发信号停止训练。
- 第 4、5 轮 train success_once 分别为 14/56（25%）、12/56（21.43%）。
  全部五轮为 17、14、4、14、12 / 56，平均 21.79%。
- 第 5 轮原生 eval：55 个唯一案例，picked=31/55，handed=11/55，success_once=10/55（18.18%），
  return=0.94545。train 成功率更高不代表 eval 已提升。
- 第 5 轮计时含 eval：1328.9 秒，其中原生 eval 526.4 秒；不含额外 eval 的训练轮仍约 13 分钟。
- `logs/20260911-02-50-00-dreamdojo_trocar_noise01/` 内已确认存在
  `dreamdojo_trocar_grpo_gr00t_n1d7/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt`、
  七个 DCP rank shard，以及 `video/eval/seed_0..6/0.mp4`。
- 未自动续训或启动下一组实验。代码整理只验证 CPU 回归/conditioning，不把这次旧进程的完成视为清理后完整 GPU 验收。

日期：2026-09-10。用户明确要求回到 RLinf 原生 train/eval，以 `dreamdojo_trocar_grpo_gr00t_n1d7.yaml` 为主配置，并将 val check 设为 5。

## 2026-09-11：停止 0.5 长训，试验 0.1 并保存 train 视频

- 用户要求停止当前实验、尝试 `noise_level=0.1`、查看 train 视频，并简化日志目录。
- 02:45 UTC 向已核实的旧主进程 PID 145195 发送 SIGTERM；随后主进程及其 Ray GPU workers 退出，八张卡均回到约 4 MiB / 0% 使用率。没有执行全局 `ray stop`、GPU reset 或删除任何旧文件。
- 保留旧运行 `logs/dreamdojo_review_20260910/native_long_v1/` 的全部记录及 `global_step_50` 等 checkpoint。停止前 TensorBoard 最后记录为 step 51，即已完成第 52 轮（train success_once=2/56）；中断时第 53 轮尚未完成。第 50 轮后的未保存更新不用于新试验。
- 主 YAML 仅改变两个有效设置：`actor.model.rl_head_config.noise_level: 0.5 -> 0.1`（原值继承自模型配置），以及 `env.train.video_cfg.save_video: false -> true`。模型通用默认值不改。
- 新试验从原始 SFT 初始化，`resume_dir` / `ckpt_path` 仍为 null；不加载旧长训权重。GPU、起点采样与混合权重、学习率、GRPO 分组、WM 35 步和 GR00T 4 步均保持不变。随机种子不改，但不将视频开关与分布式执行下的整条随机流宣称为已验证的严格配对。
- 使用原生 runner，通过 CLI `runner.max_epochs=5` 将本次限制为 5 轮，末轮做原生 eval 与 checkpoint；主 YAML 的长训上限仍为 1000。预期先检查首轮 7 个 train 视频，再根据 5 轮结果决定是否继续。
- 新日志目录使用 `logs/20260911-02-50-00-dreamdojo_trocar_noise01/`。不再在实验目录生成 `sources/`、`.diff`、`.patch`、manifest 或额外 launch-config 副本；修改说明集中在本文，实际完整配置由原生 TensorBoard logger 保存到 `tensorboard/all/config.yaml`。
- 旧源码快照属于此前审查的可选追溯材料，不是 RLinf 运行依赖。为避免丢失既有审查证据，本次不删除或搬动旧目录。

启动命令（由 shell 后台托管，不增加 Python 监督器）：

```bash
env -u CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  bash examples/embodiment/run_dreamdojo_trocar.sh \
  runner.max_epochs=5 \
  runner.logger.log_path=/localhome/local-yunl/RLinf/logs/20260911-02-50-00-dreamdojo_trocar_noise01
```

启动检查：57/57 项 CPU 回归通过（10.62 秒），Ruff lint/format 和 shell 语法检查通过；原生 Hydra 完整解析确认噪声 0.1、视频开启、5 轮上限、两侧 WM 35 步、七卡 placement。配置解析中的 MSC 插件 profile 警告与此前相同，未阻止解析成功。

02:48:59 UTC 新试验后台启动，主进程 PID / PGID / SID 为 `1477962`，随后 PPID=1；02:49:29 运行时配置确认 noise_level=0.1、train/eval save_video=true、max_epochs=5，七个模型 placement 为物理 GPU `[0,1,2,3,5,6,7]`。实际启动时间早于日志目录名称中的 02:50 标签一分钟。日志文件为 `run_embodiment.log`；未生成额外源码快照或补丁文件。

### 首轮验收（2026-09-11 03:12 UTC）

- 已完成第 1/5 轮，正在第 2 轮；主进程仍为 PID 1477962。首轮在 03:09:37 记录，TensorBoard 缓冲写盘后已可读取。
- 新旧实际 `tensorboard/all/config.yaml` 逐项比较：除 noise_level、train save_video、max_epochs 和各日志输出路径外，没有其他配置差异。
- 首轮对比（两次均从原始 SFT 采样；不是训练更新后的 eval）：

| 指标 | 旧 noise=0.5 第 1 轮 | 新 noise=0.1 第 1 轮 |
| --- | ---: | ---: |
| picked | 32/56 = 57.14% | 32/56 = 57.14% |
| handed | 8/56 = 14.29% | 19/56 = 33.93% |
| success_once | 5/56 = 8.93% | 17/56 = 30.36% |
| return | 0.80357 | 1.21429 |
| group_flat_fraction | 0/7 | 3/7 |
| group_zero_fraction | 0/7 | 2/7 |

- 首轮结果支持较强探索噪声影响后半程完成率这一假设，但不能把采样成功率增加写成 RL 学习收益；低噪声同时减少了组内奖励差异。仍需观察后续轮次与第 5 轮原生 eval。也未声称新旧运行整条随机流已做逐张量配对验证。
- 新首轮 group_return_std_mean=0.50698、grad_norm=1.25740、approx_kl=0.00258323、policy_loss=0.000511336；已记录的 50 个 scalar tag 无 NaN/Inf。梯度范数是裁剪前报告值，配置 clip_grad=1.0 未改变。
- 首轮 step=865.26 秒（不含整个模型初始化），generate_rollouts=537.45 秒，actor_training=277.44 秒；没有发现 OOM、CUDA/NCCL 或视频保存错误。
- 7 个首轮视频均位于 `video/train/seed_0/0.mp4` 至 `seed_6/0.mp4`，全部完整解码通过：121 帧、15 fps、8.07 秒、2560x960（8 个环境拼图），每文件约 3.8–4.4 MB。抽帧仅放 `/tmp/dreamdojo-noise01-video-review.HdYh63/`，不增加实验日志杂项。
- 目视抽查 seed 0/1/2/6 中首个环境的 0/2/4/6/8 秒帧：seed 1 可见物体从左侧托盘移到右侧蓝垫；其他抽查片段存在未完成转移或遮挡，局部手/物体形变与模糊仍可见。未对所有 56 条轨迹人工标注成功，亦未证明 WM 画质问题已经解决。
- TensorBoard 仍在 `http://127.0.0.1:6006/`，服务 PID 1495835，显示 `noise_05/.` 与 `noise_01/.` 两个 run；服务日志集中到 `logs/tensorboard-dreamdojo.log`。新实验目录当前只有正常的 `run_embodiment.log`、`metrics.log`、`tensorboard/`、`worker_logs/`、`video/`，checkpoint 在第 5 轮保存。

## 2026-09-10 长训配置（历史）

- 标准入口：`examples/embodiment/train_embodied_agent.py`；`run_dreamdojo_trocar.sh` 只设置本机依赖环境并转发 Hydra 参数，不改 Runner，也不启动外部 eval。
- 唯一主配置：`examples/embodiment/config/dreamdojo_trocar_grpo_gr00t_n1d7.yaml`，不合并 trial20 / long overlay。
- `cluster.component_placement.actor,env,rollout: "0-3,5-7"`。七张健康卡共同训练和评测，不预留 eval 专卡。物理 GPU 4 / UUID `GPU-77264942-9d19-703f-ff04-ee8e38ce7cb1` 仍排除。
- 56 个 train env，每卡 batch8、GRPO group8、actor global batch56 / micro batch2。沿用已验证的每卡规模，不直接扩大为每卡 24 env。
- `max_epochs: 1000`、`max_steps: -1`，不是 20 轮 trial；`val_check_interval: 5`、`save_interval: 5`。
- 原始 GR00T SFT 初始化，`resume_dir` / `ckpt_path` 为 null，不使用旧 step20，也不续训性能 smoke 的 step2。
- DreamDojo 35 步去噪；动作 30 Hz、生成视频 15 fps、20 chunks / 240 动作步 / 8 秒。固定文本缓存和零 guidance 单分支继续启用。train/eval 的 WM 均在 rollout 内常驻、外层阶段切换时卸载。
- 训练数据仍是 teleop / 30k rollout / 10k rollout 三份 train，权重 0.34 / 0.33 / 0.33。eval 使用三份 validation。

## 原生 eval 的 55 条覆盖

RLinf 要求总 eval 槽位能被 worker 数整除，七卡使用 56 个槽位。55 条真实轨迹按拼接后的数据集索引顺序固定分片，每卡 8 条；最后一个槽位复用 index 0 仅供 batch 计算。

新增 DreamDojo 专用 `eval_unique_episodes: true`：填充槽位在评测末尾不返回 done/truncation，因此原生 EnvWorker 的 newly-done 统计自动排除它，无需修改原生 Runner、通信或指标聚合。有效分母应为 `eval/num_trajectories=55`，不是 56。填充槽位仍可能出现在保存的视频中，不表示多统计一次。原生训练模式不启用此 mask。

代码核对还澄清了此前说明：旧主配置同时有 mixing weights 与 `use_fixed_reset_state_ids=true`；DreamDojo 固定 reset 分支优先，实际会按索引取样，而不是因 weights 存在就随机混合。真实问题是 56 个固定索引对 55 条数据取模，导致 index 0 重复计数。本次关闭 eval mixing weights 表达意图，并排除填充计数。

新原生协议与旧外部 full55 的 batch 分片及随机数消费不同；旧 SFT 15/55（27.27%）保留作历史参考，不能称为严格同随机协议的对照。每次原生 eval 固定起点，策略采样仍沿用原生随机数处理，不声称 checkpoint 间整条随机流完全固定。

## 追溯、检查与停止

- 修改前：`logs/dreamdojo_review_20260910/native_train_before_v1/`。
- 启动源码、SHA256 与增量补丁保存在本次运行目录；只额外覆盖唯一日志目录，不改变训练超参数。
- CPU 回归：56/56 通过（11.57 秒），`logs/dreamdojo_review_20260910/native_tests_v1.xml`。测试验证主 YAML、55 条唯一覆盖、填充成功不影响指标和训练模式不受影响。Ruff lint、启动脚本 `bash -n`、`git diff --check` 通过。
- 七卡通信检查：`logs/dreamdojo_review_20260910/native_collective7_v1/`，7/7 rank 通过，all-reduce=28、all-gather=[0,1,2,3,4,5,6]，检查进程退出码 0。
- 之前的外部评测长训草稿未启动，文件保留并标记为历史方案，不用于本次训练。其自动早停阈值也不作用于本次原生运行；不把旧评测基线套到新协议做自动早停。
- 长训练可根据原生 `eval/success_once`、`eval/return`、训练梯度/奖励等趋势提前结束，保留已经完成保存的 checkpoint。不要仅凭单轮随机训练成功率判断退化。当前不部署另一套训练/eval 监督循环。

## 运行记录

已于约 13:55 UTC 后台启动，运行目录 `logs/dreamdojo_review_20260910/native_long_v1/`。主进程 PID / PGID / SID 均为 `145195`，PPID=1，不依赖当前终端存活。实际命令为标准 RLinf 入口，配置名 `dreamdojo_trocar_grpo_gr00t_n1d7`，唯一 CLI 配置覆盖为独立日志路径。

13:56 UTC 已核对运行日志：配置校验通过，actor/env/rollout 均生成七个 placement，物理硬件 ranks 为 `[0,1,2,3,5,6,7]`；TensorBoard 聚合事件与实际配置已创建，模型／worker 正在初始化。节点元数据探测日志可能显示八张可见卡，不等于把训练 worker 分配到 GPU 4。

启动快照为本目录 `sources/` / `manifest.json`，解析配置为 `launch_config.yaml`；运行时校验后的配置为 `tensorboard/all/config.yaml`。`changes_since_before.patch` 对比选定文件快照；启动 shell 此前不在快照清单中，本次补丁中以完整文件保存，不声称它原先不存在。

首次周期 eval 在第 5 轮后，不额外改为每轮评测。这里尚未声称首轮更新完成或已经获得新的 eval 成功率。

```bash
env -u CUDA_VISIBLE_DEVICES bash examples/embodiment/run_dreamdojo_trocar.sh \
  runner.logger.log_path=/localhome/local-yunl/RLinf/logs/dreamdojo_review_20260910/native_long_v1
```

本次后台启动会将 stdout/stderr 写入运行目录的 `console.log`。原生 `worker_logs/`、`tensorboard/all/`、`video/eval/` 和 checkpoint 子目录用于排查与评估。
