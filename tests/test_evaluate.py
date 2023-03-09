import numpy as np
import networkx as nx
from accutrack.evaluate import detection_test, get_node_matching_map, get_vertex_errors


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
    mtrix_dict = {0: {"det": mtrix, "comp_ids": comp_ids, "gt_ids": gt_ids}}
    matching = get_node_matching_map(mtrix_dict)
    assert matching == [(12, 3), (15, 3), (14, 7)]


def test_get_node_matching_map():
    comp_ids = [3, 7, 10]
    gt_ids = [4, 12, 14, 15]
    mtrix = np.zeros((3, 4), dtype=np.uint8)
    mtrix[0, 1] = 1
    mtrix[0, 3] = 1
    mtrix[1, 2] = 1
    mtrix_dict = {0: {"det": mtrix, "comp_ids": comp_ids, "gt_ids": gt_ids}}
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
        0: {"det": mtrix, "comp_ids": comp_ids, "gt_ids": gt_ids},
        1: {"det": mtrix2, "comp_ids": comp_ids, "gt_ids": gt_ids},
    }
    matching = get_node_matching_map(mtrix_dict)
    assert matching == [(12, 3), (15, 3), (14, 7), (12, 7), (4, 10)]


def test_get_vertex_errors():
    comp_ids = [3, 7, 10]
    comp_ids_2 = list(np.asarray(comp_ids) + 1)
    gt_ids = [4, 12, 14, 17]
    gt_ids_2 = list(np.asarray(gt_ids) + 1)

    mtrix = np.zeros((3, 4), dtype=np.uint8)
    mtrix[0, 1] = 1
    mtrix[0, 3] = 1
    mtrix[1, 2] = 1

    mtrix2 = np.zeros((3, 4), dtype=np.uint8)
    mtrix2[1, 1] = 1
    mtrix2[2, 0] = 1
    mtrix_dict = {
        0: {"det": mtrix, "comp_ids": comp_ids, "gt_ids": gt_ids},
        1: {"det": mtrix2, "comp_ids": comp_ids_2, "gt_ids": gt_ids_2},
    }
    gt_g = nx.DiGraph()
    gt_g.add_nodes_from(gt_ids + gt_ids_2)
    comp_g = nx.DiGraph()
    comp_g.add_nodes_from(comp_ids + comp_ids_2)

    vertex_errors = get_vertex_errors(gt_g, comp_g, mtrix_dict)
    assert vertex_errors["ns"] == 1
    assert vertex_errors["tp"] == 3
    assert vertex_errors["fp"] == 2
    assert vertex_errors["fn"] == 3

    assert gt_g.nodes[15]["is_fn"]
    assert comp_g.nodes[3]["is_ns"]
    assert comp_g.nodes[7]["is_tp"]
    assert comp_g.nodes[10]["is_fp"]


# test_get_vertex_errors()
