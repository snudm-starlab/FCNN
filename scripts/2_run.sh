GPU=2

cd ..
CUDA_VISIBLE_DEVICES=$GPU python train.py -net dmobilenetv2 -gpu 
