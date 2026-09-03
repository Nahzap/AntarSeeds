"""Unit tests for detector SOD and instance metrics."""



import numpy as np



from src.training.detector_metrics import (

    det_precision_recall_f1,

    mask_iou,

    sod_f_measure,

    sod_mae,

)





def test_sod_mae_identical_maps():

    m = np.ones((32, 32), dtype=np.float32) * 0.7

    assert sod_mae(m, m) == 0.0





def test_sod_mae_opposite():

    pred = np.zeros((16, 16), dtype=np.float32)

    gt = np.ones((16, 16), dtype=np.float32)

    assert abs(sod_mae(pred, gt) - 1.0) < 1e-6





def test_sod_f_measure_perfect():

    pred = np.zeros((32, 32), dtype=np.float32)

    pred[8:24, 8:24] = 1.0

    gt = pred.copy()

    assert sod_f_measure(pred, gt) > 0.99





def test_sod_f_measure_empty_gt_empty_pred():

    pred = np.zeros((16, 16), dtype=np.float32)

    gt = np.zeros((16, 16), dtype=np.float32)

    assert sod_f_measure(pred, gt) == 1.0





def test_mask_iou_perfect():

    m = np.zeros((10, 10), dtype=np.float32)

    m[2:8, 2:8] = 1.0

    assert mask_iou(m, m) == 1.0





def test_det_f1_perfect_match():

    boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]

    r = det_precision_recall_f1(boxes, boxes)

    assert r["det_f1"] == 1.0

    assert r["det_precision"] == 1.0

    assert r["det_recall"] == 1.0





def test_det_f1_no_predictions():

    r = det_precision_recall_f1([], [(0, 0, 10, 10)])

    assert r["det_recall"] == 0.0

    assert r["det_precision"] == 1.0


