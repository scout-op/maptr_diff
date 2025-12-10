#!/usr/bin/env bash
# Distributed training script for MapTR
# Usage: bash tools/dist_train.sh configs/maptr_av2_example.py 8

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
