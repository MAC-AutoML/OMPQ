#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "mobilenetv2" \
 --path "/Path/to/Base_model" \
 --dataset "imagenet" \
 --save_path '/Path/to/Dataset/' \
 --beta 3.3 \
 --model_size 0.9 \
 --quant_type "PTQ"