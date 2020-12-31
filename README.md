# Stronger and Faster Wasserstein Adversarial Attacks 

Code for [Stronger and Faster Wasserstein Adversarial Attacks][paper], appeared in ICML 2020. This repository contains the implementation of our Wasserstein adversarial attacks and pretrained robust models.
The implementation of the projection operator and the linear minimization oracle for Wasserstein constraint can be of independent interest.

[paper]: https://arxiv.org/abs/2008.02883


## Instructions for running the code
Dependency: PyTorch 1.5.1 with CUDA 10.2, scipy 1.5.0, and advertorch 0.2.3

Before running the procedure, it is required to install the sparse tensor package:
```
cd sparse_tensor
python setup.py install
```
The sparse tensor package includes several functions for initialization of sparse tensors.

Checkout the following bash scripts for different attack methods:
```
bash bash/test_sinkhorn.sh             # projected Sinkhorn
bash bash/test_projected_gradient.sh   # PGD with dual projection
bash bash/test_frank_wolfe.sh          # Frank-Wolfe with dual LMO
```
You may want to switch to the option `download=True` in the Line 111 and 124 in `data.py` for the *first* run.

The folder `./checkpoints` stores all pretrained models. The names of the checkpoints indicate their training methods. For examples, `mnist_vanilla.pth` and `mnist_adv_training.pth` are pretrained  models directly taken from Wong et al., 2019. `mnist_adv_training_attack-frank_eps-0.3.pth` is a model adversarially trained by Frank-Wolfe using epsilon=0.3.

Checkout the following bash script for adversarial training using Frank-Wolfe:
```
bash bash/train.sh
```

Checkout the following bash script for model clean accuracy and lp adversarial attacks:
```
bash bash/test.sh
```
