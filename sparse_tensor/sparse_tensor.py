import torch

import sparse_tensor_cpp as sparse


def initialize_dense_cost(height, width):
    """Return a tensor of size (height * width, height * width)"""
    cost = sparse.initialize_dense_cost_cpp(height, width)
    return cost


def initialize_sparse_cost(height, width, kernel_size, inf):
    """
    Return two tensors
    indices: two dimensional tensor of size (2, height * width * kernel_size ** 2)
    values: one dimensional tensor of size (height * width * kernel_size ** 2, )
    """
    indices, values = sparse.initialize_sparse_cost_cpp(height, width, kernel_size, inf)
    return indices, values


def initialize_sparse_coupling(X, kernel_size):
    """
    Return two tensors
    indices: two dimensional tensor of size (4, batch_size * channel * img_size * kernel_size ** 2)
    values: one dimensional tensor of size (batch_size * channel * img_size * kernel_size ** 2, )
    """
    indices, values = sparse.initialize_sparse_coupling_cpp(X, kernel_size)
    return indices, values


if __name__ == "__main__":
    device = "cuda"

    print("initializing a dense cost matrix for 2 by 2 image...")
    print(initialize_dense_cost(2, 2))

    print("initializing a sparse cost matrix for 3 by 3 image...")
    print(initialize_sparse_cost(3, 3, 3, 100))
    # print(initialize_sparse_cost(3, 3, 3).to_dense())

    print("initializing a sparse coupling matrix for 3 by 3 image...")
    X = torch.randn((2, 1, 3, 3), dtype=torch.float)
    print(X.size())
    print(initialize_sparse_coupling(X.view(2, 1, 3, 3), 1).to_dense())
