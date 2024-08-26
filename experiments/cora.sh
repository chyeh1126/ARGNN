#!/bin/bash

for seed in {10..14}
do
    python train_ARGNN_feature.py \
        --seed ${seed} \
        --dataset cora \
        --attack_structure no \
        --attack_feature no \
        --ptb_rate 0.15 \
        --alpha 3 \
        --sigma 100 \
        --beta 0.3 \
        --eta 1 \
        --t_small 0.1 \
        --lr 1e-3 \
        --lr_adj 1e-3 \
        --epoch 1000 \
        --label_rate 0.01 \
        --threshold 0.6 \
        --n_p 1000 \
        --gnnlayers 8 \
        --linlayers 1 \
        --upth_st 0.011 \
        --upth_ed 0.001 \
        --lowth_st 0.1 \
        --lowth_ed 0.5 \
        --val_rate 0.1 \
        --test_rate 0.8
done