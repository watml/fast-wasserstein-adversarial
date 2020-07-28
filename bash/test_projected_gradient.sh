#!/bin/bash

dataset=mnist
ckpt=mnist_vanilla
# ckpt=mnist_robust
# ckpt=mnist_adv_training
# ckpt=mnist_adv_training_attack-frank_eps-0.3
batch=100
num_batch=1
eps=0.5
lr=0.1
nb_iter=300
postprocess=1

# dataset=cifar
# # ckpt=cifar_vanilla
# # ckpt=cifar_adv_training
# ckpt=cifar_adv_training_attack-frank_eps-0.005
# batch=100
# num_batch=1
# eps=0.005
# lr=0.01
# nb_iter=300
# postprocess=1

# dataset=imagenet
# ckpt=imagenet_resnet50
# batch=20
# num_batch=1
# eps=0.001
# lr=0.01
# nb_iter=20
# postprocess=1

python projected_gradient.py --dataset $dataset \
                             --checkpoint $ckpt \
                             --batch_size $batch \
                             --num_batch $num_batch \
                             --eps $eps \
                             --kernel_size 5 \
                             --lr $lr \
                             --nb_iter $nb_iter \
                             --dual_max_iter 25 \
                             --grad_tol 1e-4 \
                             --int_tol 1e-4 \
                             --seed 0 \
                             --postprocess $postprocess \
                             --save_img_loc ./adversarial_examples/projected_gradient.pt \
                             --save_info_loc None
