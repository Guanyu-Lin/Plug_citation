#!/bin/bash
DATASET=$1
PlugD=$2
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20089 \
    train.py -c config/Downstream/PlugD/${DATASET}PlugD.config \
    -g 0 \
    --checkpoint ${PlugD} \
    2>&1 | tee log/Downstream/${DATASET}-plugd-large.log
