GR00T with the Cosmos-Nano World Model
======================================================================

Train GR00T N1.7 with GRPO inside an action-conditioned Cosmos-Nano simulator
for H2 medical-pad unfolding. Start with your SFT policy, trained world model,
frame classifier and LeRobot episodes, then evaluate and resume through the
native RLinf entry points. The world model and classifier stay frozen during RL.

Overview
--------

The policy commands joints, Cosmos generates the next images, and a task-stage
classifier scores each generated frame.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      GR00T N1.7

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Medical-pad unfolding

   .. grid-item-card:: Hardware
      :text-align: center

      8 NVIDIA H200 GPUs

Tasks
~~~~~

Use the episode split and quality labels exported with your dataset.

.. list-table::
   :header-rows: 1

   * - Task
     - Reset Data
     - Reference Split
   * - H2 medical-pad unfolding
     - LeRobot v2.1 RGB, states and commands
     - 554 train / 61 validation episodes

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each action chunk advances 16 frames at 30 Hz. The world model predicts RGB;
the last commanded target supplies the next policy's joint-state input.

.. list-table::
   :header-rows: 1

   * - Field
     - Representation
   * - Observation
     - Left-head RGB at 480 × 640 and 58 joints, ordered as 7 left-arm,
       7 right-arm, 22 left-hand and 22 right-hand joints.
   * - Action
     - 16 absolute 58-joint targets in physical units. The Cosmos codec
       converts them to relative actions using the preceding command;
       frame-zero resets use the measured initial state as the anchor.
   * - Reward
     - ResNet18 expected score over five classes with values [-2, -1, 0, 1, 2].
       Training uses consecutive score differences, including the reset score.
   * - Prompt
     - The task description stored in the episode metadata.

An episode fails after 30 consecutive negative predictions; switching between
-2 and -1 keeps the failure streak. It succeeds after 30 consecutive predictions
of 2. If the 480-frame limit arrives first, a final valid prediction of 2 also
counts as success. Other timeouts remain unsuccessful. The reset image does
not count toward either streak, and frames after termination are padding.

Installation
------------

Use the CUDA runtime and H2-capable Cosmos source snapshot that produced your
world-model checkpoint. It must provide ``cosmos_framework.inference`` and
``cosmos_framework.utils.h2_action_codec``. The GR00T source must provide
``gr00t.model.gr00t_n1d7``. The adapter was exercised with the combined
CUDA 13 / PyTorch 2.10 runtime on H200 GPUs.

``docker/cosmos3/Dockerfile`` layers RLinf and GR00T onto an existing Cosmos
base image. Its build context contains source directories named ``RLinf``,
``Isaac-GR00T`` and ``cosmos-framework-private``. Pin these source snapshots
and supply the base image explicitly:

.. code-block:: bash

   docker build -f RLinf/docker/cosmos3/Dockerfile \
       --build-arg BASE_IMAGE=your-cosmos-h2-image:tested \
       -t rlinf-cosmos-h2:local .

The installer constrains the additional packages to the compiled versions
already present in that image. Run the following commands inside this runtime.

Download the Model
------------------

Place your trained assets on disk and point the adapter at them. Use the world
model's original action statistics: matching joint names alone does not establish
matching normalization. The backend checks the statistics checksum stored in
the world-model training configuration.

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

The dataset supplies ``meta/info.json`` and ``meta/split.json``. Each split row
has ``episode_index``, ``split`` (``train`` or ``val``) and ``quality``.
Camera aliases in ``meta/modality.json`` resolve to
``observation.images.cam_left_high``. Both parquet joint-name lists must match
the WM statistics. Classifier resizing, normalization and class values come
from its inference configuration; the current five-class model uses FP32
bilinear preprocessing, ``pre_resize: null`` and ``channels_last: true``.

Run It
------

The default recipe uses learning rate ``3e-6``, flow noise ``0.5``, 150 updates,
64 training environments, GRPO groups of 8 and two rollout epochs. Training
and evaluation each use 30 world-model diffusion steps. One 8-GPU worker
allocation provides 8 training environments per GPU.

.. code-block:: bash

   bash examples/embodiment/run_cosmos3_medical_pads.sh \
       runner.logger.log_path=/path/to/run

This starts fresh from SFT. Reset sampling gives success episodes half the
probability and all three imperfect categories together the other half;
episodes within each pool are uniform. To give each of the four categories
one quarter, add:

.. code-block:: bash

   env.train.balanced_qualities=[success,imperfect1,imperfect2,imperfect3]

For the other current learning-rate/noise setting, add
``actor.optim.lr=1e-5 actor.model.rl_head_config.noise_level=0.3``.
All members of a GRPO group share the reset episode. The ordinary evaluation
path uses the policy's evaluation mode and does not apply training flow noise.

Evaluate SFT before training, or load an exported RL actor for the same resets:

.. code-block:: bash

   bash examples/embodiment/run_cosmos3_medical_pads.sh --eval \
       runner.logger.log_path=/path/to/sft-eval \
       env.eval.evaluation_result_dir=/path/to/sft-eval/episodes

   bash examples/embodiment/run_cosmos3_medical_pads.sh --eval \
       runner.ckpt_path=/path/to/global_step_35/actor/model_state_dict/full_weights.pt \
       runner.logger.log_path=/path/to/rl-eval \
       env.eval.evaluation_result_dir=/path/to/rl-eval/episodes

Both commands use the validation split and fixed episode order. For the reference
61 episodes on 8 workers, 64 slots exclude three padding episodes from the
metrics. For another dataset size, set ``env.eval.total_num_envs`` to the smallest
multiple of the worker count that covers that split. Resume training with
``runner.resume_dir=/path/to/checkpoints/global_step_N`` and the desired total
``runner.max_steps``; this restores the native distributed checkpoint.

Visualization and Results
-------------------------

TensorBoard and videos are written under ``runner.logger.log_path``. Watch
``eval/success_final`` for terminal-rule success. The per-episode JSONL files
include episode IDs, final labels, failure, timeout and termination frame.

.. list-table::
   :header-rows: 1

   * - Diagnostic
     - Meaning
   * - ``group_reward_std_mean``
     - Spread of trajectory returns within GRPO groups.
   * - ``group_reward_equal_fraction``
     - Fraction of groups with return range at most 1e-6.
   * - ``group_mixed_fraction``
     - Groups straddling the return threshold; this is a return-based
       diagnostic, not a per-episode physical success measurement.

Check the adapter without loading model assets:

.. code-block:: bash

   OMP_NUM_THREADS=1 pytest tests/unit_tests/test_cosmos3.py

These tests cover decoding and action anchors, H2 action conversion, classifier
preprocessing, group sampling, video frames, terminal padding and timeout rules.
They exercise the actual adapter against a fake external Cosmos SDK. World-model
classifier scores and success rates require video review and real-robot checks
to assess simulator contact errors.
