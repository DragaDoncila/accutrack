import numpy as np
from accutrack.evaluate import detection_test, get_node_matching_map


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

def test_get_node_matching_map():
    comp_ids = [3, 7, 10]
    gt_ids = [4, 12, 14, 15]
    mtrix = np.zeros((3, 4), dtype=np.uint8)
    mtrix[0, 1] = 1
    mtrix[0, 3] = 1
    mtrix[1, 2] = 1
    mtrix_dict = {
        0: {
            'det': mtrix,
            'comp_ids': comp_ids,
            'gt_ids': gt_ids
        }
    }
    matching = get_node_matching_map(mtrix_dict)
    assert matching == [(12, 3), (15, 3), (14, 7)]

def test_get_node_matching_map():
    comp_ids = [3, 7, 10]
    gt_ids = [4, 12, 14, 15]
    mtrix = np.zeros((3, 4), dtype=np.uint8)
    mtrix[0, 1] = 1
    mtrix[0, 3] = 1
    mtrix[1, 2] = 1
    mtrix_dict = {
        0: {
            'det': mtrix,
            'comp_ids': comp_ids,
            'gt_ids': gt_ids
        }
    }
    matching = get_node_matching_map(mtrix_dict)
    assert matching == [(12, 3), (15, 3), (14, 7)]

def test_get_node_matching_map_multiple_frames():
    comp_ids = [3, 7, 10]
    gt_ids = [4, 12, 14, 15]
    mtrix = np.zeros((3, 4), dtype=np.uint8)
    mtrix[0, 1] = 1
    mtrix[0, 3] = 1
    mtrix[1, 2] = 1

    mtrix2 = np.zeros((3, 4), dtype=np.uint8)
    mtrix2[1, 1] = 1
    mtrix2[2, 0] = 1
    mtrix_dict = {
        0: {
            'det': mtrix,
            'comp_ids': comp_ids,
            'gt_ids': gt_ids
        },
        1: {
            'det': mtrix2,
            'comp_ids': comp_ids,
            'gt_ids': gt_ids            
        }
    }
    matching = get_node_matching_map(mtrix_dict)
    assert matching == [(12, 3), (15, 3), (14, 7), (12, 7), (4, 10)]         

