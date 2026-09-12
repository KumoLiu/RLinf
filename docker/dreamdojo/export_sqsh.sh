#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 2 || "$2" != /*.sqsh ]]; then
    echo "Usage: bash export_sqsh.sh IMAGE /absolute/new/image.sqsh" >&2
    exit 2
fi
IMAGE="$1"
OUTPUT="$2"
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }
command -v enroot >/dev/null
sudo -n docker image inspect "${IMAGE}" >/dev/null
mkdir -p "$(dirname "${OUTPUT}")"
sudo -n env ENROOT_MAX_PROCESSORS=16 \
    ENROOT_CACHE_PATH="$(dirname "${OUTPUT}")/enroot-cache" \
    ENROOT_SQUASH_OPTIONS='-comp zstd -Xcompression-level 3 -noappend' \
    enroot import --output "${OUTPUT}" "dockerd://${IMAGE}"
sudo -n chown "$(id -u):$(id -g)" "${OUTPUT}"
unsquashfs -s "${OUTPUT}"
sha256sum "${OUTPUT}" | tee "${OUTPUT}.sha256"
