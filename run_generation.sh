#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

fname="e108079799d746809f898619890b6606"
seg_mode="click"

# # deocclusion
# python scene_deocc.py \
# --image_path data/others/${fname}.png \
# --output_path outputs/openset/${fname} \
# --threshold 0.3 \
# --deocc_mode flux \
# --seg_mode ${seg_mode} \


# pose estimation inference
python scene_generation.py \
--image_path data/others/${fname}.png \
--output_path outputs/${fname} \
--pose_model ckpts/SceneMaker_openset_ckpts \
--depth_mode moge \
--use_direct_pose \
--threshold 0.3 \
--pcd_mode pcd2 \
--deocc_mode flux \
--seg_mode ${seg_mode} \
--num_objs 4 \

# ckpts/SceneMaker_indoor_ckpts
# ckpts/SceneMaker_openset_ckpts

