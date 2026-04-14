# config='./configs/train-geometry-diffusion/step1x-3d-geometry-1300m.yaml'
config='./configs/train-geometry-diffusion/step1x-3d-geometry-label-1300m.yaml'
export CUDA_VISIBLE_DEVICES=0
python train.py --config $config --train --gpu 0 

# python train.py --config $config --train --gpu 0 system.use_lora=True # for lora training

# multi-GPU training
# torchrun train.py \
#     --config $config \
#     --train \
#     --gpu 0,1,2,3,4,5,6,7 \
#     trainer.num_nodes=$num_nodes \
#     system.use_lora=True \
#     --use_ema \