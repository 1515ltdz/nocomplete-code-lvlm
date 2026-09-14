#!/usr/bin/env bash
set -euo pipefail

SESSION="lead_train"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/qwenvl/train/train_lead.py"
OUTDIR="${SCRIPT_DIR}/checkpoints"
LOG_PATH="${OUTDIR}/train_launch.log"

mkdir -p "${OUTDIR}"
tmux new-session -d -s "${SESSION}" "bash -lc 'CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29652 ${SCRIPT} |& tee ${LOG_PATH}; echo TRAIN_EXIT_CODE:\$?; exec bash'"
tmux set-option -t "${SESSION}" remain-on-exit on

echo "SESSION=${SESSION}"
echo "OUTDIR=${OUTDIR}"
echo "LOG_PATH=${LOG_PATH}"
