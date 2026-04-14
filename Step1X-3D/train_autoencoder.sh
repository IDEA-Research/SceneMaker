config='./configs/train-geometry-autoencoder/michelangelo.yaml'

export CUDA_VISIBLE_DEVICES=0
python train.py --config $config --train --gpu 0