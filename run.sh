CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=20089 train.py -c config/Downstream/PlugD/ASQAPlugD_test.config -g 4 --checkpoint None
