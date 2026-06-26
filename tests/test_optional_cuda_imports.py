import pytest
import torch


def test_model_package_imports_without_chamfer_extension():
    import openpoints.models  # noqa: F401


def test_chamfer_module_imports_without_chamfer_extension():
    import openpoints.cpp.chamfer_dist as chamfer_dist

    assert chamfer_dist.ChamferDistanceL1 is not None


def test_chamfer_missing_extension_raises_import_error_when_used():
    import openpoints.cpp.chamfer_dist as chamfer_dist

    if chamfer_dist._chamfer is not None:
        pytest.skip("chamfer extension is installed in this environment")

    loss = chamfer_dist.ChamferDistanceL1()
    with pytest.raises(ImportError, match="chamfer"):
        loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))


def test_emd_module_imports_without_emd_cuda_extension():
    from openpoints.cpp.emd import emd

    assert emd is not None


def test_emd_missing_extension_raises_import_error_when_used():
    import importlib

    emd_module = importlib.import_module("openpoints.cpp.emd.emd")
    from openpoints.cpp.emd import emd

    if emd_module._emd_cuda is not None:
        pytest.skip("emd_cuda extension is installed in this environment")

    loss = emd()
    with pytest.raises(ImportError, match="emd_cuda"):
        loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))


def test_pointops_imports_without_pointops_cuda_extension():
    from openpoints.cpp.pointops.functions import pointops

    assert pointops.furthestsampling is not None

    if pointops.pointops_cuda.__class__.__name__ != "_MissingCudaExtension":
        pytest.skip("pointops_cuda extension is installed in this environment")

    with pytest.raises(ImportError, match="pointops_cuda"):
        pointops.pointops_cuda.furthestsampling_cuda


def test_pointnet2_missing_extension_raises_import_error_when_used():
    from openpoints.cpp import pointnet2_cuda

    if pointnet2_cuda.__class__.__name__ != "_MissingCudaExtension":
        pytest.skip("pointnet2_batch_cuda extension is installed in this environment")

    with pytest.raises(ImportError, match="pointnet2_batch_cuda"):
        pointnet2_cuda.three_nn_wrapper
