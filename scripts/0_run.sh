GPU=0

cd ..
CUDA_VISIBLE_DEVICES=$GPU python train.py -net mobilenetv2 -gpu
