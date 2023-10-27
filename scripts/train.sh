GPU=0
NET=fcnn34
RHO=6
NU=2
cd src/
CUDA_VISIBLE_DEVICES=$GPU python train.py -net $NET -gpu -rho $RHO -nu $NU
