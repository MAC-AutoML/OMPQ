#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet18" \
 --path "/Path/to/Basemodel" \
 --dataset "imagenet" \
 --save_path '/Path/to/Dataset' \
 --beta 100 \
 --model_size 3.0 \
 --quant_type "PTQ"
