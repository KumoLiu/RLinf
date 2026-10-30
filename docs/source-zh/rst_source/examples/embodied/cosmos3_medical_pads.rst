在 Cosmos-Nano 世界模型中训练 GR00T
======================================================================

本页介绍如何在 Cosmos-Nano 神经模拟器中，用 GRPO 训练 H2 医用垫展开任务的 GR00T N1.7 policy。准备好 SFT policy、世界模型、帧分类器和 LeRobot 数据后，可通过 RLinf 原生入口训练、评估和续训。RL 期间只更新 policy，世界模型与分类器保持冻结。

概览
----

Policy 输出关节指令，Cosmos 生成后续图像，任务阶段分类器再逐帧计算奖励。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T N1.7

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      医用垫展开

   .. grid-item-card:: 硬件
      :text-align: center

      8 张 NVIDIA H200 GPU

任务
~~~~

重置状态和质量类别都来自数据集导出的 episode 清单。

.. list-table::
   :header-rows: 1

   * - 任务
     - 重置数据
     - 参考划分
   * - H2 医用垫展开
     - LeRobot v2.1 图像、关节状态与指令
     - 554 条训练 / 61 条验证 episode

观测与动作
~~~~~~~~~~

每个动作块包含 16 帧，频率为 30 Hz。世界模型只预测 RGB 图像；下一次 policy 输入的关节状态使用上一条目标指令作为近似。

.. list-table::
   :header-rows: 1

   * - 字段
     - 表示
   * - Observation
     - 480 × 640 左头相机 RGB，以及按左臂 7、右臂 7、左手 22、右手 22 排列的 58 维关节状态。
   * - Action
     - 16 条物理单位下的 58 维绝对关节目标。Cosmos codec 以上一条指令为基准转换成相对动作；首帧重置时以实测初始状态为基准。
   * - Reward
     - ResNet18 对 [-2, -1, 0, 1, 2] 五类分值计算期望分。训练奖励为相邻帧期望分之差，第一步也减去重置图像的期望分。
   * - Prompt
     - Episode 元数据中的任务描述。

连续 30 帧预测为负分则失败；-2 与 -1 交替出现时仍累计失败帧数。连续 30 帧预测为 2 则成功。若先到达 480 帧上限，最后一个有效帧预测为 2 也算成功，其他 timeout 不算成功。重置图像不计入连续帧数，结束后的补齐帧不参与训练。

安装
----

使用训练该世界模型时的 CUDA 环境与支持 H2 的 Cosmos 源码快照。源码需包含 ``cosmos_framework.inference`` 和 ``cosmos_framework.utils.h2_action_codec``；GR00T 源码需包含 ``gr00t.model.gr00t_n1d7``。该适配在 H200 的 CUDA 13 / PyTorch 2.10 组合环境中运行过。

``docker/cosmos3/Dockerfile`` 在现有 Cosmos 基础镜像上安装 RLinf 和 GR00T。构建上下文包含 ``RLinf``、``Isaac-GR00T`` 和 ``cosmos-framework-private`` 三个源码目录。固定各目录的源码版本，并显式指定基础镜像：

.. code-block:: bash

   docker build -f RLinf/docker/cosmos3/Dockerfile \
       --build-arg BASE_IMAGE=your-cosmos-h2-image:tested \
       -t rlinf-cosmos-h2:local .

安装脚本会约束新增依赖，使其沿用基础镜像中已编译组件的版本。后续命令均在这一运行环境内执行。

下载模型
--------

将模型和数据放到本地，再通过环境变量指定路径。动作归一化必须使用世界模型原来的统计文件，仅核对关节名称还不足以保证归一化一致。Backend 会检查世界模型训练配置中的统计文件校验和。

.. code-block:: bash

   export COSMOS3_ROOT=/path/to/h2-capable-cosmos-source
   export GR00T_ROOT=/path/to/Isaac-GR00T
   export RLINF_PYTHON=python
   export GR00T_MODEL_PATH=/path/to/sft-checkpoint
   export GR00T_BACKBONE_PATH=/path/to/Cosmos-Reason2-2B
   export COSMOS3_WM_CHECKPOINT=/path/to/cosmos/iter_000020000
   export COSMOS3_WM_CONFIG=/path/to/cosmos/config.yaml
   export COSMOS3_ACTION_STATS=/path/to/cosmos/action_stats.json
   export H2_DATASET_PATH=/path/to/lerobot-dataset
   export CLASSIFIER_REWARD_CHECKPOINT=/path/to/classifier/reward_model.pt
   export CLASSIFIER_REWARD_CONFIG=/path/to/classifier/inference_config.yaml
   export COSMOS3_GPUS=0-7

数据集包含 ``meta/info.json`` 和 ``meta/split.json``。划分清单的每行包含 ``episode_index``、``split`` （``train`` 或 ``val``）和 ``quality``。``meta/modality.json`` 中的相机别名需指向 ``observation.images.cam_left_high``，parquet 中状态与动作的关节名称顺序都须匹配 WM 统计文件。分类器的缩放、归一化与分值来自 inference 配置；当前五分类模型使用 FP32 双线性缩放，配置为 ``pre_resize: null``、``channels_last: true``。

运行
----

默认配置使用 ``3e-6`` 学习率、``0.5`` flow noise，训练 150 代。64 个环境按每组 8 个执行 GRPO，每代两个 rollout epoch；训练和评估都使用 30 步世界模型扩散采样。8 卡配置下，每张卡运行 8 个训练环境。

.. code-block:: bash

   bash examples/embodiment/run_cosmos3_medical_pads.sh \
       runner.logger.log_path=/path/to/run

该命令从 SFT 权重开始训练。初始状态按成功与失败各占一半的概率采样，三个 imperfect 类别合并为失败池，池内按 episode 均匀采样。若要让四类各占四分之一，在命令末尾添加：

.. code-block:: bash

   env.train.balanced_qualities=[success,imperfect1,imperfect2,imperfect3]

另一组当前使用的学习率和噪声参数为 ``actor.optim.lr=1e-5 actor.model.rl_head_config.noise_level=0.3``。同一 GRPO 组共用初始 episode。常规评估使用 policy 的 eval 模式，不施加训练用的 flow noise。

训练前可先评估 SFT，再用同一批初态评估导出的 RL actor：

.. code-block:: bash

   bash examples/embodiment/run_cosmos3_medical_pads.sh --eval \
       runner.logger.log_path=/path/to/sft-eval \
       env.eval.evaluation_result_dir=/path/to/sft-eval/episodes

   bash examples/embodiment/run_cosmos3_medical_pads.sh --eval \
       runner.ckpt_path=/path/to/global_step_35/actor/model_state_dict/full_weights.pt \
       runner.logger.log_path=/path/to/rl-eval \
       env.eval.evaluation_result_dir=/path/to/rl-eval/episodes

这两个命令使用验证集的固定 episode 顺序。参考数据在 8 个 worker 上用 64 个槽位覆盖 61 条验证数据，补齐的 3 条不参与指标计算。更换数据规模后，将 ``env.eval.total_num_envs`` 设为覆盖验证集所需的最小 worker 数整数倍。续训使用 ``runner.resume_dir=/path/to/checkpoints/global_step_N`` 和目标总代数 ``runner.max_steps``，恢复 RLinf 原生分布式 checkpoint。

可视化与结果
------------

TensorBoard 和视频保存在 ``runner.logger.log_path`` 下。``eval/success_final`` 表示按结束规则计算的成功率；逐 episode 的 JSONL 记录包含编号、末帧类别、failure、timeout 和结束帧。

.. list-table::
   :header-rows: 1

   * - 诊断指标
     - 含义
   * - ``group_reward_std_mean``
     - GRPO 组内轨迹回报的离散程度。
   * - ``group_reward_equal_fraction``
     - 组内回报极差不超过 1e-6 的组所占比例。
   * - ``group_mixed_fraction``
     - 回报分布在阈值两侧的组所占比例；它是基于回报的诊断值，不等同于真机任务成功率。

以下测试可以在不加载模型资产的情况下检查适配逻辑：

.. code-block:: bash

   OMP_NUM_THREADS=1 pytest tests/unit_tests/test_cosmos3.py

测试覆盖解码与动作基准、H2 动作转换、分类器预处理、组采样、视频帧、终止后补齐和 timeout 规则，使用实际适配代码与模拟的外部 Cosmos SDK。WM 分类得分及成功率仍需结合视频复核和真机测试，检查模拟器是否存在接触误差。
