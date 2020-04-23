#include<cassert>

#include<cmath>
#include<tuple>
#include<vector>

#include<torch/extension.h>

#include<iostream>
using namespace std;

#define _img_index(i, j) ((i) * width + (j))
#define square(a) ((a) * (a))
#define dist(i1, i2, j1, j2) (sqrt(square((i1) - (i2)) + square((j1) - (j2))))


tuple<torch::Tensor, torch::Tensor> initialize_sparse_cost_cpp(int height, int width, int kernel_size, int inf)
{
    assert(kernel_size % 2 == 1);

    torch::Tensor indices = torch::zeros({2, height * width * square(kernel_size)},
                                         torch::dtype(torch::kInt64));
    auto indices_a = indices.accessor<int64_t, 2>();

    torch::Tensor values = torch::zeros({height * width * square(kernel_size)},
                                        torch::dtype(torch::kFloat32));
    auto values_a = values.accessor<float, 1>();

    int kernel_range = kernel_size / 2;

    int index = 0;

    for(int h1 = 0; h1 < height; h1++)
        for(int w1 = 0; w1 < width; w1++)
            for(int k1 = 0; k1 < kernel_size; k1++)
                for(int k2 = 0; k2 < kernel_size; k2++) {
                    int h2 = h1 + k1 - kernel_range;
                    int w2 = w1 + k2 - kernel_range;

                    indices_a[0][index] = _img_index(h1, w1);
                    indices_a[1][index] = _img_index((h2 + height) % height, (w2 + width) % width);
   
                    if ((0 <= h2 && h2 < height) && (0 <= w2 && w2 < width))
                        values_a[index] = dist(h1, h2, w1, w2);
                    else 
                        values_a[index] = inf;

                    index++;
                }

    assert(index == height * width * square(kernel_size));

    return make_tuple(indices, values);
}

    
tuple<torch::Tensor, torch::Tensor> initialize_sparse_coupling_cpp(torch::Tensor X, int kernel_size)
{
    assert(kernel_size % 2 == 1);

    // CAUTION: assume that X is a CPU tensor
    auto X_a = X.accessor<float, 4>();

    int batch_size = X_a.size(0);
    int channel = X_a.size(1);
    int height = X_a.size(2);
    int width = X_a.size(3);

    torch::Tensor indices = torch::zeros({4, batch_size * channel * height * width * square(kernel_size)},
                                         torch::dtype(torch::kInt64));
    auto indices_a = indices.accessor<int64_t, 2>();
   
    torch::Tensor values = torch::zeros(batch_size * channel * height * width * square(kernel_size),
                                        torch::dtype(torch::kFloat32));
    auto values_a = values.accessor<float, 1>();
   
    int kernel_range = kernel_size / 2;

    int index = 0;

    for(int b = 0; b < batch_size; b++)
        for(int c = 0; c < channel; c++) {
            for(int h1 = 0; h1 < height; h1++)
                for(int w1 = 0; w1 < width; w1++)
                    for(int k1 = 0; k1 < kernel_size; k1++)
                        for(int k2 = 0; k2 < kernel_size; k2++) {
                            int h2 = h1 + k1 - kernel_range;
                            int w2 = w1 + k2 - kernel_range;

                            indices_a[0][index] = b;
                            indices_a[1][index] = c;
                            indices_a[2][index] = _img_index(h1, w1);
                            indices_a[3][index] = _img_index((h2 + height) % height, (w2 + width) % width);

                            if (h2 == h1 && w1 == w2)
                                values_a[index] = X_a[b][c][h1][w1];
                            else 
                                values_a[index] = 0;

                            index++;
                        }
        }

    assert(index == batch_size * channel * height * width * square(kernel_size));

    return make_tuple(indices, values);
}


torch::Tensor initialize_dense_cost_cpp(int height, int width)
{
    torch::Tensor cost = torch::zeros({height * width, height * width});
    auto cost_accessor = cost.accessor<float, 2>();

    for(int i1 = 0; i1 < height; i1++) 
        for(int j1 = 0; j1 < width; j1++)
            for(int i2 = 0; i2 < height; i2++)
                for(int j2 = 0; j2 < width; j2++)
                    cost_accessor[_img_index(i1, j1)][_img_index(i2, j2)] = dist(i1, i2, j1, j2);

    return cost;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize_dense_cost_cpp", &initialize_dense_cost_cpp, "");
    m.def("initialize_sparse_cost_cpp", &initialize_sparse_cost_cpp, "");
    m.def("initialize_sparse_coupling_cpp", &initialize_sparse_coupling_cpp, "");
}
