import numpy as np
from accutrack.evaluate import detection_test, get_frame_det_test_matrix


def test_detection_test_true():
    gt_blob = np.zeros(shape=(10, 10), dtype=np.uint8)
    gt_blob[2:6, 2:6] = 1

    comp_blob = np.zeros(shape=(10, 10), dtype=np.uint8)
    comp_blob[3:6, 3:6] = 1

    assert detection_test(gt_blob, comp_blob)

def test_detection_test_false():
    gt_blob = np.zeros(shape=(10, 10), dtype=np.uint8)
    gt_blob[1:6, 1:6] = 1

    comp_blob = np.zeros(shape=(10, 10), dtype=np.uint8)
    comp_blob[4:6, 4:6] = 1   
    
    assert not detection_test(gt_blob, comp_blob)

def test_detection_test_boundary():
    "Test fails if computed blob covers exactly half of gt blob's pixels"
    gt_blob = np.zeros(shape=(10, 10), dtype=np.uint8)
    gt_blob[1:2, 1:2] = 1

    comp_blob = np.zeros(shape=(10, 10), dtype=np.uint8)
    comp_blob[2, 2] = 1

    assert not detection_test(gt_blob, comp_blob)
