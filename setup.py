import numpy
from setuptools import Extension, setup
try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
except:
        raise ModuleNotFoundError("Please install pytorch >= 1.1 before proceeding.")

setup(
    ext_modules=[
        CUDAExtension(
            name="openpoints.chamfer_cuda",  # as it would be imported
            sources=["cuda/chamfer/chamfer.cu",
                     "cuda/chamfer/chamfer_cuda.cpp"], # all sources are compiled into a single binary file
        ),
        CUDAExtension(
            name="openpoints.cuda.emd",
            sources=["cuda/emd/cuda/emd.cpp",
                     "cuda/emd/cuda/emd_kernel.cu"]
        ),
        CUDAExtension(
            name="opentpoints.cuda.pointnet2",
            sources=[
                "cuda/pointnet2/src/ball_query.cpp",
                "cuda/pointnet2/src/ball_query_gpu.cu",
                "cuda/pointnet2/src/group_points.cpp",
                "cuda/pointnet2/src/group_points_gpu.cu",
                "cuda/pointnet2/src/interpolate.cpp",
                "cuda/pointnet2/src/interpolate_gpu.cu",
                "cuda/pointnet2/src/pointnet2_api.cpp",
                "cuda/pointnet2/src/sampling.cpp",
                "cuda/pointnet2/src/sampling_gpu.cu",
            ]
        ),
        CUDAExtension(
            name="openpoints.cuda.pointops",
            sources=[
                "cuda/pointops/src/pointops_api.cpp",
                "cuda/pointops/src/aggregation/aggregation_cuda.cpp",
                "cuda/pointops/src/aggregation/aggregation_cuda_kernel.cu",
                "cuda/pointops/src/ballquery/ballquery_cuda.cpp",
                "cuda/pointops/src/ballquery/ballquery_cuda_kernel.cu",
                "cuda/pointops/src/grouping/grouping_cuda.cpp",
                "cuda/pointops/src/grouping/grouping_cuda_kernel.cu",
                "cuda/pointops/src/interpolation/interpolation_cuda.cpp",
                "cuda/pointops/src/interpolation/interpolation_cuda_kernel.cu",
                "cuda/pointops/src/knnquery/knnquery_cuda.cpp",
                "cuda/pointops/src/knnquery/knnquery_cuda_kernel.cu",
                "cuda/pointops/src/sampling/sampling_cuda.cpp",
                "cuda/pointops/src/sampling/sampling_cuda_kernel.cu",
                "cuda/pointops/src/subtraction/subtraction_cuda.cpp",
                "cuda/pointops/src/subtraction/subtraction_cuda_kernel.cu",
            ]
        ),
        Extension(
            name="openpoints.cuda.subsampling",
            sources=[
                "cuda/subsampling/wrapper.cpp",
                "cuda/subsampling/cpp_utils/cloud/cloud.cpp",
                "cuda/subsampling/grid_subsampling/grid_subsampling.cpp",
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[
        numpy.get_include(),
        "cuda/subsampling/cpp_utils/nanoflann/",
        "cuda/subsampling/cpp_utils/cloud/",
        "cuda/subsampling/grid_subsampling/",
        "cuda/pointnet2/src/",
        "cuda/pointops/src/sampling",
        "cuda/pointops/src/grouping",
        "cuda/pointops/src",
        "cuda/pointops/src/knnquery",
        "cuda/pointops/src/ballquery",
        "cuda/pointops/src/interpolation",
        "cuda/pointops/src/subtraction",
        "cuda/pointops/src/aggregation",
    ]
)
