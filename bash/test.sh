#!/bin/bash

# dataset=mnist
# ckpt=mnist_vanilla
# ckpt=mnist_adv_training
# ckpt=mnist_adv_training_attack-frank_eps-0.3

dataset=cifar
# ckpt=cifar_vanilla
# ckpt=cifar_adv_training
ckpt=cifar_adv_training_attack-frank_eps-0.005

# dataset=imagenet
# ckpt=imagenet_resnet50

python test.py --dataset $dataset \
               --checkpoint $ckpt \
               --batch_size 100 \
               --num_batch 100 \
               --norm None \
               --eps 10 \
               --lr 1 \
               --nb_iter 1000 \
               --seed 0 \
               --save_img_loc None
