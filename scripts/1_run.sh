GPU=1
EXP_NAME=test
cd ..
CUDA_VISIBLE_DEVICES=$GPU python train.py -net dfmobilenetv2 -gpu -rho 3 -nu 2 -exp $EXP_NAME
# CUDA_VISIBLE_DEVICES=$GPU python train.py -net fcnn34 -gpu -rho 4 -nu 2
