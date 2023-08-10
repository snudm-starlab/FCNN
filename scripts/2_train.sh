cd ..
CUDA_VISIBLE_DEVICES=2 python ttt.py -net fresnet34 -gpu -lr 0.1 -kappa 4 -nu 4 -alpha 0.7 -tau 10
