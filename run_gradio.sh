#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=7

python scenemaker_app.py