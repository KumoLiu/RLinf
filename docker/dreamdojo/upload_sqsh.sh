#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 || ! -f "$1" || "$1" != *.sqsh ]]; then
    echo "Usage: bash upload_sqsh.sh /path/image.sqsh" >&2
    exit 2
fi
IMAGE="$1"
REMOTE="${REMOTE:-nvidia-cluster}"
REMOTE_DIR="${REMOTE_DIR:-/lustre/fsw/portfolios/healthcareeng/users/yunl/docker}"
DEST="${REMOTE_DIR}/$(basename "${IMAGE}")"
[[ "${DEST}" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo "Unsupported destination" >&2; exit 2; }
SSH=(ssh -T -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UpdateHostKeys=no)
LOCAL_SHA="$(sha256sum "${IMAGE}" | awk '{print $1}')"
if "${SSH[@]}" "${REMOTE}" "test -e '${DEST}'"; then
    REMOTE_SHA="$("${SSH[@]}" "${REMOTE}" "sha256sum '${DEST}'" | awk '{print $1}')"
    [[ "${LOCAL_SHA}" == "${REMOTE_SHA}" ]] || { echo "Refusing to replace existing image" >&2; exit 1; }
    echo "Already uploaded and verified: ${REMOTE}:${DEST}"
    exit 0
fi
"${SSH[@]}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"
rsync -avh --partial --append-verify --info=progress2 \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UpdateHostKeys=no' \
    "${IMAGE}" "${REMOTE}:${DEST}.partial"
REMOTE_SHA="$("${SSH[@]}" "${REMOTE}" "sha256sum '${DEST}.partial'" | awk '{print $1}')"
[[ "${LOCAL_SHA}" == "${REMOTE_SHA}" ]] || { echo "Checksum mismatch; partial retained" >&2; exit 1; }
"${SSH[@]}" "${REMOTE}" "test ! -e '${DEST}' && mv '${DEST}.partial' '${DEST}'"
echo "Verified SHA256 ${LOCAL_SHA}: ${REMOTE}:${DEST}"
