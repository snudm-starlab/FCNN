GPU=2
NET=resnet34
RHO=6
NU=2
cd src/
CUDA_VISIBLE_DEVICES=$GPU python train.py -net $NET -gpu -rho $RHO -nu $NU \
                       -alpha 0.9 -tau 10
