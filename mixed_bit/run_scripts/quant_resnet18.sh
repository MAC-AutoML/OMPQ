#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet18" \
 --path "Exp_base/resnet18_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --beta 100 \
 --model_size 3.0 \
 --quant_type "PTQ"
