#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# # mixed indoor data
# torchrun --nproc_per_node=8 --master_port 29501 train.py \
# --config configs/image-to-scene-diffusion/indoor-mixed-dinov2reglarge518-pixartflow.yaml \
# --train --gpu 0,1,2,3,4,5,6,7 \

# openset data
torchrun --nproc_per_node=8 --master_port 29501 train.py \
--config configs/image-to-scene-diffusion/openset-dinov2reglarge518-pixartflow.yaml \
--train --gpu 0,1,2,3,4,5,6,7 \
