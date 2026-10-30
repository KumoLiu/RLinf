#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export PATH="/opt/cosmos-venv/bin:/usr/local/bin:$PATH"
export UV_LINK_MODE=copy LD_LIBRARY_PATH='' NO_ALBUMENTATIONS_UPDATE=1
# Protect the tested base image's compiled CUDA packages and model libraries.
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
names = ['torch', 'torchvision', 'numpy', 'transformers', 'diffusers', 'peft',
         'flash-attn', 'transformer-engine', 'natten', 'ray', 'av', 'pyarrow']
Path('/tmp/cosmos-base-constraints.txt').write_text(
    ''.join(f'{name}=={version(name)}\n' for name in names))
PY
uv pip install --python /opt/cosmos-venv/bin/python \
    -c /tmp/cosmos-base-constraints.txt -r /opt/src/RLinf/requirements/embodied/cosmos3_h2.txt
uv pip install --python /opt/cosmos-venv/bin/python --no-deps --no-sources -e /opt/src/RLinf
# GR00T's Python/torch/NumPy metadata describes its standalone environment.
# Validate its source with Cosmos' compiled stack instead of re-resolving it.
python -m pip install --no-deps --no-build-isolation --ignore-requires-python -e /opt/src/Isaac-GR00T
export PYTHONPATH=/opt/src/RLinf:/opt/src/Isaac-GR00T:/opt/src/cosmos-framework-private
python /opt/src/RLinf/docker/cosmos3/check_runtime.py imports
uv pip freeze --python /opt/cosmos-venv/bin/python > /opt/rlinf-cosmos3-packages.txt
touch /opt/rlinf-cosmos3-build-complete
uv cache clean
