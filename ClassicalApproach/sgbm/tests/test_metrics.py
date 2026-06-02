import numpy as np
from ..eval.metrics import per_image_metrics


def test_perfect_prediction():
    gt = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32)
    pred = gt.copy()
    m = per_image_metrics(pred, gt)
    assert m["abs_rel"] == 0.0
    assert m["rmse"] == 0.0
    assert m["delta_1_25"] == 1.0
    assert m["coverage"] == 1.0
    assert m["n_valid_pred"] == 4
    assert m["n_valid_gt"] == 4


def test_no_prediction():
    gt = np.array([[1.0, 2.0]], dtype=np.float32)
    pred = np.zeros_like(gt)
    m = per_image_metrics(pred, gt)
    assert m["coverage"] == 0.0
    assert m["n_valid_pred"] == 0
    assert np.isnan(m["abs_rel"])


def test_constant_offset():
    gt = np.array([10.0, 20.0, 0.0], dtype=np.float32)
    pred = np.array([11.0, 22.0, 0.0], dtype=np.float32)
    m = per_image_metrics(pred, gt)
    assert abs(m["abs_rel"] - 0.1) < 1e-6
    assert m["delta_1_25"] == 1.0
