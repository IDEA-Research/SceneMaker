#!/bin/bash
#SBATCH -J train3d
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:hgx:1
#SBATCH --qos=preemptive

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python inference.py