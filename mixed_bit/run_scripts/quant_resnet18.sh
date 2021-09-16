#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet18" \
 --path "/Path/to/Base_model" \
 --dataset "imagenet" \
 --save_path '/Path/to/Dataset/' \
 --beta 10.0 \
 --model_size 6.7 \
 --quant_type "QAT"