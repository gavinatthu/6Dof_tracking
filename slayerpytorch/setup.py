from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='slayerCuda',
    ext_modules=[
        CUDAExtension(
            name='slayerCuda',
            sources=[
                'src/cuda/slayerKernels.cu'
            ],
            depends=[
                'src/cuda/spikeKernels.h',
                'src/cuda/convKernels.h',
                'src/cuda/shiftKernels.h'
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-arch=sm_60', '-O3', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
