GPU=0
NET=resnet34
RHO=6
NU=2
WEIGHTS=checkpoint/$NET/teacher.pth
cd src/
CUDA_VISIBLE_DEVICES=$GPU python test.py -net $NET -gpu -rho $RHO -nu $NU -weights $WEIGHTS
