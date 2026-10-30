# syntax=docker/dockerfile:1
# Context: a directory containing the three pinned source snapshots.
# Build the base with cosmos-framework-private/docker/h2.Dockerfile first.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY RLinf /opt/src/RLinf
COPY Isaac-GR00T /opt/src/Isaac-GR00T
COPY cosmos-framework-private /opt/src/cosmos-framework-private
RUN bash /opt/src/RLinf/docker/cosmos3/install.sh
ENV PATH=/opt/cosmos-venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONPATH=/opt/src/RLinf:/opt/src/Isaac-GR00T:/opt/src/cosmos-framework-private \
    COSMOS3_ROOT=/opt/src/cosmos-framework-private \
    GR00T_ROOT=/opt/src/Isaac-GR00T \
    RLINF_PYTHON=/opt/cosmos-venv/bin/python \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/src/RLinf
CMD ["/bin/bash"]
