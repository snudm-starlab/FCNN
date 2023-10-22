MODEL=fcnn34
cd ..
# CUDA_VISIBLE_DEVICE=1 python test.py -net $MODEL -weights checkpoint/$MODEL/Wednesday_30_August_2023_09h_03m_51s/conv3dresnet34-174-best.pth
CUDA_VISIBLE_DEVICE=1 python test.py -net $MODEL -weights checkpoint/$MODEL/best_7525.pth
