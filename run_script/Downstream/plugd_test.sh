#!/bin/bash
DATASET=$1
PlugD=$2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=20088 \
    train.py -c config/Downstream/PlugD/${DATASET}PlugD_test.config --only_eval\
    -g 0 \
    --checkpoint ${PlugD} \
    2>&1 | tee log/Downstream/${DATASET}-plugd-large-test.log
