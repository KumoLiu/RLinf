# syntax=docker/dockerfile:1
# Independent reproduction of the locally validated RLinf/DreamDojo runtime.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
FROM ${BASE_IMAGE} AS embodied-dreamdojo-gr00t-image
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /usr/local/bin/uv
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON=3.10.21
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs ffmpeg libgl1 libglib2.0-0 libegl1 \
    libsm6 libxext6 libxrender1 build-essential linux-libc-dev pkg-config
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.10.21 && uv venv --python 3.10.21 --seed /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt cuda-wheels.txt /opt/rlinf-build/
# Reproduce installed versions without re-resolving conflicting upstream pins.
# Install headless OpenCV last: both cv2 distributions existed on the source host.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
      -r /opt/rlinf-build/requirements.txt -r /opt/rlinf-build/cuda-wheels.txt && \
    uv pip install --python /opt/venv/bin/python --no-deps --reinstall \
      opencv-python-headless==4.12.0.88
COPY src/ /opt/src/
COPY source-manifest.json runtime-versions.json check_runtime.py /opt/rlinf-build/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps --no-sources \
      -e /opt/src/RLinf -e /opt/src/DreamDojo \
      -e /opt/src/DreamDojo/packages/cosmos-oss \
      -e /opt/src/DreamDojo/packages/cosmos-cuda && \
    python -m pip install --no-deps --no-build-isolation --ignore-requires-python \
      -e /opt/src/Isaac-GR00T
ENV PYTHONPATH=/opt/src/RLinf:/opt/src/DreamDojo \
    DREAMDOJO_ROOT=/opt/src/DreamDojo \
    EMBODIED_PATH=/opt/src/RLinf/examples/embodiment \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
    TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /opt/src/RLinf
RUN python /opt/rlinf-build/check_runtime.py imports
# No custom entrypoint: Pyxis/Slurm and Docker can invoke native RLinf commands.
CMD ["/bin/bash"]
