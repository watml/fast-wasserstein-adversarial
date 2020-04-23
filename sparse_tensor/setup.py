from setuptools import setup
from torch.utils import cpp_extension

setup(name='sparse_tensor_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_tensor_cpp', ['sparse_tensor.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
