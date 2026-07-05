from pathlib import Path


def test_pyproject_declares_current_package_layout():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "openpoints"' in pyproject
    assert 'package-dir = { "openpoints" = "." }' in pyproject
    assert '"openpoints.cpp.chamfer_dist"' in pyproject
    assert '"openpoints.dataset.semantic_kitti"' in pyproject

    for package_dir in ("cpp", "dataset", "models", "transforms", "utils"):
        assert (root / package_dir / "__init__.py").is_file()
