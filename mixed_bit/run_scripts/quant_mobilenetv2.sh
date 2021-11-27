#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "mobilenetv2" \
 --path "Exp_base/mobilenetv2_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --beta 0.000000001 \
 --model_size 0.9 \
 --quant_type "PTQ"