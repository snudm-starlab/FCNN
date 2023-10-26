GPU=2
cd ..
for A in 0.9
do
    for T in 10
    do
        CUDA_VISIBLE_DEVICES=$GPU python train.py -net conv3dresnet34 -gpu -lr 0.1 -rho 6 -nu 2 -alpha $A -tau $T
    done
done

