from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name="depth_renderer",
    ext_modules=[
        CUDAExtension(
            name='depth_renderer_cuda',
            sources=[
                'gpu/depth_renderer_cuda.cpp',
                'gpu/depth_renderer.cu'
            ]
        ),
        CppExtension('depth_renderer', [
            'cpp/depth_renderer.cpp'
        ])
    ],
    cmdclass = {
        'build_ext': BuildExtension
    }
)
