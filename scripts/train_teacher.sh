GPU=2
NET=resnet34
cd src/
CUDA_VISIBLE_DEVICES=$GPU python train.py -net $NET -gpu
