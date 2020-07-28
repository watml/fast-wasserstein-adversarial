#!/bin/bash

dataset=mnist
ckpt=mnist_vanilla
# ckpt=mnist_adv_training
# ckpt=mnist_adv_training_attack-frank_eps-0.3
batch=100
num_batch=1
eps=0.5
lr=0.1
nb_iter=300
lam=1000

# dataset=cifar
# ckpt=cifar_vanilla
# ckpt=cifar_adv_training
# ckpt=cifar_adv_training_attack-frank_eps-0.005
# batch=100
# num_batch=1
# eps=0.005
# lr=0.1
# nb_iter=300
# lam=3000

# dataset=imagenet
# ckpt=imagenet_resnet50
# batch=100
# num_batch=1
# eps=0.001
# lr=0.1
# nb_iter=10
# lam=100000

# lam is the inverse of gamma in the paper, i.e., gamma = 1 / lam

python sinkhorn.py --dataset $dataset \
                   --checkpoint $ckpt \
                   --batch_size $batch \
                   --num_batch $num_batch \
                   --eps $eps \
                   --kernel_size 5 \
                   --lr $lr \
                   --nb_iter $nb_iter \
                   --lam $lam \
                   --sinkhorn_max_iter 500 \
                   --stop_abs 1e-4 \
                   --stop_rel 1e-4 \
                   --seed 0 \
                   --save_img_loc ./adversarial_examples/sinkhorn.pt \
                   --save_info_loc None
