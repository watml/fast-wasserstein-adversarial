# Fast Computation of Wasserstein Adversarial Examples

## Instructions for running the code
Dependency: PyTorch 1.5.1 with CUDA 10.2, scipy 1.5.0, and advertorch

To run the procedure, it is required to install the sparse tensor package by the following:
```
cd sparse_tensor
python setup.py install
```
The sparse tensor package includes several functions for initialization of sparse tensors.

Run the following bash scripts:
```
bash test_sinkhorn.sh             # projected Sinkhorn
bash test_projected_gradient.sh   # PGD with dual projection
bash test_frank_wolfe.sh          # Frank-Wolfe with dual LMO
```
