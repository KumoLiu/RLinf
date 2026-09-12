# Actor micro batch 最小性能检查（2026-09-11）

目的：比较 micro batch 2 / 4 / 8 是否能完成非零梯度的原生 actor 更新，以及耗时、显存。
正式 YAML 默认值不变；不启动长训练，不修改原生 train/eval，不保存 source/.diff 快照。

## 测试边界

- 物理 GPU 0、1、2、3、5、6、7；7 卡 full-shard FSDP、BF16、gradient checkpointing。
- global batch 56、每卡 8 条样本；micro 2 / 4 / 8 分别累积 4 / 2 / 1 次再更新。
- 使用当前升级后的 GR00T SFT base；每个 rank 读取 teleop train 对应 episode 的第 2 帧及 state，
  复制 8 次。通过模型原生 `predict_action_batch(mode="train")` 生成独立 Flow-SDE chains，
  将这一批完整 forward inputs 固定下来供三组复用。GR00T 降噪 4 步、noise level 0.1。
- **advantages 是人为设置的零均值正负序列**，用于确保反向传播不是零梯度空测；
  不使用 WM 或 reward classifier，不产生 success once，也不用于判断能否学会任务。
- 复用 `FSDPModelManager.setup_model_and_optimizer`、
  `EmbodiedFSDPActor.train_micro_batch` 和 `optimizer_step`，没有另写 loss 或去噪实现。
  只替代 Ray 调度外壳；不包含 WM、rollout worker 常驻开销或 CPU offload 阶段。
- 每组预热一次、测量三次；每次恢复相同参数、AdamW 状态和 RNG。
  计时含 CPU→GPU 输入、前向/反向、原生 optimizer 前 empty_cache、optimizer step；
  不含模型加载、固定输入生成、状态恢复和结果校验。
- 用每次七个 rank 中最慢的更新时间比较吞吐；显存为 PyTorch 峰值 allocated/reserved，
  不是整张卡总占用。检查 finite/nonzero grad norm、实际参数变化以及采样梯度方向。

`num-env=7` 不适用于当前 7 卡、每组 8 条轨迹的完整 GRPO rollout；该基准根本不启动 env。
若未来测完整单卡 pipeline，最小可用 8 env，但它的 FSDP 显存和通信不能代表 7 卡。

## 复现

在 RLinf 根目录、七张健康卡空闲时执行（换机器需调整设备与 global batch）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=7 \
  toolkits/world_model/dreamdojo_actor_perf.py \
  --output logs/20260911-actor-microbench/new-run --repeats 3
```

工具拒绝已存在的输出目录，结果包括 settings.json、rank_N.json、results.json。
CPU 分批校验：`python -m pytest tests/unit_tests/test_dreamdojo_actor_perf.py`。

## 实测结果

本机七张 H200 NVL；PyTorch 2.7.0+cu128，GR00T 源码
`2d9a9fce9811d10510d1263344fbf5253f33639d`。2026-09-11 10:21 UTC 完成，进程退出码 0。
产物：`logs/20260911-actor-microbench/run01/`，旁边的 `run01.console.log` 为启动日志。

| micro batch | 累积次数 | 56 样本/更新，中位耗时 | 相对 micro=2 吞吐 | 峰值 allocated | 峰值 reserved |
| --- | --- | --- | --- | --- | --- |
| 2 | 4 | 12.501 秒 | 1.00× | 10.00 GiB | 16.93 GiB |
| 4 | 2 | 6.852 秒 | 1.82× | 11.74 GiB | 19.41 GiB |
| 8 | 1 | 4.004 秒 | 3.12× | 12.23 GiB | 19.30 GiB |

耗时先取每次七个 rank 的最大值，再取三次中位数；显存取七卡、三次测量的最大值。
reserved 是缓存分配器预留值，受碎片影响，因此 micro=8 不一定高于 micro=4。

- 三组共 12 个全局更新（各 1 次预热 + 3 次测量），七个 rank 全部通过有限、非零梯度及实际参数更新检查。
- grad norm：2 为 90.2598–90.2613；4 为 90.2593–90.2637；8 为 90.2552–90.2611。
- micro=4、8 与 micro=2 首次预热的采样梯度最小 cosine 分别为 0.999884、0.999950。
  这是各 rank 梯度抽样方向检查，不是逐元素或真实任务效果完全等价的证明。
- rank 0 的未按累积次数缩放的 loss 分别约 0.01001644、0.01001549、0.01001656。
  原生 `actor/total_loss` 在单次 micro backward 中已除以累积次数，不能直接拿该字段跨 micro 比较；
  本工具额外记录 `loss_unscaled`（累积各 micro 的 scaled loss）。
- 7 个 CPU 分批单元测试通过，Ruff lint/format 通过。

结论：**actor 单独更新时，micro=8 是本次三个候选中更好的选择**，耗时比 2 少约 68%，
allocated 只增加约 2.23 GiB。4 耗时少约 45%。没有证据表明此处必须限制为 2。
新增的仅是性能工具、分批测试及本记录；本次未修改正式 YAML、loss、原生 train/eval 或模型实现。

下一步建议用 CLI `actor.micro_batch_size=8` 跑一次原生完整 pipeline 短验证，确认 WM/rollout
offload 后的实际共置显存以及完整轮次时间，再用于长训练。当前测试不含 WM 推理、offload 或真实奖励，
所以 **3.12× 只属于 actor 更新部分，不能当作整代训练的加速比，也不能据此推断 success once 会提升**。
