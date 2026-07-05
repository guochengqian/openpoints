from pathlib import Path


def test_pointmlp_exposes_training_loss_hooks():
    source = (Path(__file__).resolve().parents[1] / "models" / "backbone" / "pointmlp.py").read_text()

    assert "criterion_args=None" in source
    assert "self.criterion = build_criterion_from_cfg(criterion_args)" in source
    assert "def get_loss(self, pred, gt, inputs=None):" in source
    assert "def get_logits_loss(self, data, gt):" in source
