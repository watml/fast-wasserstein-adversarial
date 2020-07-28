# Stronger and Faster Wasserstein Adversarial Attacks 

## Instructions for running the code
Dependency: PyTorch 1.5.1 with CUDA 10.2, scipy 1.5.0, and advertorch 0.2.3

Before running the procedure, it is required to install the sparse tensor package:
```
cd sparse_tensor
python setup.py install
```
The sparse tensor package includes several functions for initialization of sparse tensors.

Checkout the following bash scripts for detailed arguments:
```
bash bash/test_sinkhorn.sh             # projected Sinkhorn
bash bash/test_projected_gradient.sh   # PGD with dual projection
bash bash/test_frank_wolfe.sh          # Frank-Wolfe with dual LMO
```

The folder `./checkpoints` stores all pretrained models. The names of the checkpoints indicate their training method. For examples, `mnist_vanilla.pth` and `mnist_adv_training.pth` are pretrained  models directly taken from Wong et al., 2019. `mnist_adv_training_attack-frank_eps-0.3.pth` is a model adversarially trained by Frank-Wolfe using epsilon=0.3.