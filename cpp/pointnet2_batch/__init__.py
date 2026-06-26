"""pointnet2_batch CUDA extension loader.

The PyPI wheel ships Python sources only. Build this extension from
``cpp/pointnet2_batch`` for full CUDA training/evaluation.
"""


class _MissingCudaExtension:
    def __init__(self, module_name, error):
        self.module_name = module_name
        self.error = error

    def __getattr__(self, name):
        raise ImportError(
            f"{self.module_name} is not installed. Build OpenPoints CUDA ops from "
            "source first, e.g. `cd cpp/pointnet2_batch && python setup.py install`, "
            "or install a wheel that includes the compiled CUDA extension."
        ) from self.error


try:
    import pointnet2_batch_cuda as pointnet2_cuda
except ImportError as exc:
    pointnet2_cuda = _MissingCudaExtension("pointnet2_batch_cuda", exc)
