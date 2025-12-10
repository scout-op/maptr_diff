#!/usr/bin/env bash
# Distributed testing script for MapTR
# Usage: bash tools/dist_test.sh configs/maptr_av2_example.py checkpoints/latest.pth 8

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch ${@:4}
