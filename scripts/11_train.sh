cd ..
CUDA_VISIBLE_DEVICES=1 python ttt.py -net fresnetv2_34 -gpu -lr 0.1 -kappa 8 -nu 8 -alpha 0.7 -tau 10
