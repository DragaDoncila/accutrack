import numpy as np
import networkx as nx
import pytest
from accutrack.evaluate import (
    assign_edge_errors,
    detection_test,
    get_comp_subgraph,
    get_ctc_tra_error,
    get_edge_error_counts,
    get_node_matching_map,
    get_vertex_errors,
    get_weighted_edge_error_sum,
    get_weighted_error_sum,
    get_weighted_vertex_error_sum,
)


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
    assert not gt_g.nodes[17]["is_fn"]

    assert comp_g.nodes[3]["is_ns"]
    assert not comp_g.nodes[7]["is_ns"]

    assert comp_g.nodes[7]["is_tp"]
    assert not comp_g.nodes[3]["is_tp"]

    assert comp_g.nodes[10]["is_fp"]
    assert not comp_g.nodes[7]["is_fp"]


def test_get_comp_subgraph():
    comp_ids = [3, 7, 10]
    comp_ids_2 = list(np.asarray(comp_ids) + 1)

    comp_g = nx.DiGraph()
    comp_g.add_nodes_from(comp_ids + comp_ids_2)
    nx.set_node_attributes(comp_g, False, "is_tp")
    comp_g.nodes[7]["is_tp"] = True
    comp_g.nodes[8]["is_tp"] = True
    comp_g.nodes[11]["is_tp"] = True
    comp_g.add_edge(3, 4)
    comp_g.add_edge(7, 11)

    induced_graph = get_comp_subgraph(comp_g)
    assert sorted(induced_graph.nodes) == [7, 8, 11]
    assert list(induced_graph.edges) == [(7, 11)]


def test_assign_edge_errors():
    comp_ids = [3, 7, 10]
    comp_ids_2 = list(np.asarray(comp_ids) + 1)
    comp_ids += comp_ids_2

    gt_ids = [4, 12, 17]
    gt_ids_2 = list(np.asarray(gt_ids) + 1)
    gt_ids += gt_ids_2
    mapping = [(4, 3), (12, 7), (17, 10), (5, 4), (18, 11), (13, 8)]

    # need a tp, fp, fn
    comp_edges = [(3, 4), (7, 8)]
    comp_g = nx.DiGraph()
    comp_g.add_nodes_from(comp_ids)
    comp_g.add_edges_from(comp_edges)
    nx.set_node_attributes(comp_g, True, "is_tp")
    nx.set_edge_attributes(comp_g, 0, "is_parent")

    gt_edges = [(4, 5), (17, 18)]
    gt_g = nx.DiGraph()
    gt_g.add_nodes_from(gt_ids)
    gt_g.add_edges_from(gt_edges)
    nx.set_edge_attributes(gt_g, 0, "is_parent")
    assign_edge_errors(gt_g, comp_g, mapping)

    assert comp_g.edges[(3, 4)]["is_tp"]
    assert comp_g.edges[(7, 8)]["is_fp"]
    assert gt_g.edges[(17, 18)]["is_fn"]


def test_assign_edge_errors_semantics():
    comp_ids = [3, 7, 10]
    comp_ids_2 = list(np.asarray(comp_ids) + 1)
    comp_ids += comp_ids_2

    gt_ids = [4, 12, 17]
    gt_ids_2 = list(np.asarray(gt_ids) + 1)
    gt_ids += gt_ids_2
    mapping = [(4, 3), (12, 7), (17, 10), (5, 4), (18, 11), (13, 8)]

    # need a tp, fp, fn
    comp_edges = [(3, 4)]
    comp_g = nx.DiGraph()
    comp_g.add_nodes_from(comp_ids)
    comp_g.add_edges_from(comp_edges)
    nx.set_node_attributes(comp_g, True, "is_tp")
    nx.set_edge_attributes(comp_g, 0, "is_parent")

    gt_edges = [(4, 5), (17, 18)]
    gt_g = nx.DiGraph()
    gt_g.add_nodes_from(gt_ids)
    gt_g.add_edges_from(gt_edges)
    nx.set_edge_attributes(gt_g, 0, "is_parent")
    gt_g.edges[(4, 5)]["is_parent"] = 1
    assign_edge_errors(gt_g, comp_g, mapping)

    assert comp_g.edges[(3, 4)]["is_wrong_semantic"]
    assert not comp_g.edges[(3, 4)]["is_tp"]


def test_count_edge_errors():
    comp_ids = [3, 7, 10]
    comp_ids_2 = list(np.asarray(comp_ids) + 1)
    comp_ids += comp_ids_2

    gt_ids = [4, 12, 17]
    gt_ids_2 = list(np.asarray(gt_ids) + 1)
    gt_ids += gt_ids_2
    mapping = [(4, 3), (12, 7), (17, 10), (5, 4), (18, 11), (13, 8)]

    # need a tp, fp, fn
    comp_edges = [(3, 4), (7, 8)]
    comp_g = nx.DiGraph()
    comp_g.add_nodes_from(comp_ids)
    comp_g.add_edges_from(comp_edges)
    nx.set_node_attributes(comp_g, True, "is_tp")
    nx.set_edge_attributes(comp_g, 0, "is_parent")

    gt_edges = [(4, 5), (17, 18)]
    gt_g = nx.DiGraph()
    gt_g.add_nodes_from(gt_ids)
    gt_g.add_edges_from(gt_edges)
    nx.set_edge_attributes(gt_g, 0, "is_parent")
    assign_edge_errors(gt_g, comp_g, mapping)
    edge_errors = get_edge_error_counts(gt_g, comp_g)
    assert edge_errors["tp"] == 1
    assert edge_errors["fp"] == 1
    assert edge_errors["fn"] == 1
    assert edge_errors["ws"] == 0


def test_weighted_edge_sum():
    edge_error_counts = {"fp": 5, "fn": 12, "ws": 2}
    error_sum = get_weighted_edge_error_sum(edge_error_counts)
    assert error_sum == 19

    error_sum = get_weighted_edge_error_sum(edge_error_counts, 2, 2, 2)
    assert error_sum == 38

    error_sum = get_weighted_edge_error_sum(edge_error_counts, 0.5)
    assert error_sum == 16.5

    error_sum = get_weighted_edge_error_sum(edge_error_counts, 0.5, 2, 0.2)
    assert error_sum == 26.9

    error_sum = get_weighted_edge_error_sum(edge_error_counts, 0, 0, 0)
    assert error_sum == 0


def test_weighted_vertex_sum():
    vertex_error_counts = {"fp": 12, "fn": 4, "ns": 6}

    error_sum = get_weighted_vertex_error_sum(vertex_error_counts)
    assert error_sum == 22

    error_sum = get_weighted_vertex_error_sum(vertex_error_counts, 0.5)
    assert error_sum == 19

    error_sum = get_weighted_vertex_error_sum(vertex_error_counts, 0.5, 0.2, 0.8)
    assert pytest.approx(error_sum) == 8.6

    error_sum = get_weighted_vertex_error_sum(vertex_error_counts, 0, 0, 0)
    assert error_sum == 0


def test_weighted_error_sum():
    edge_error_counts = {"fp": 5, "fn": 12, "ws": 2}
    vertex_error_counts = {"fp": 12, "fn": 4, "ns": 7}

    error_sum = get_weighted_error_sum(vertex_error_counts, edge_error_counts)
    assert error_sum == 42

    error_sum = get_weighted_error_sum(
        vertex_error_counts, edge_error_counts, vertex_ns_weight=0, edge_ws_weight=0
    )
    assert error_sum == 33

    error_sum = get_weighted_error_sum(
        vertex_error_counts, edge_error_counts, 0.5, 0.5, 0.5
    )
    assert error_sum == 30.5


def test_ctc_tra_measure():
    gt_graph = nx.DiGraph()
    gt_graph.add_nodes_from(list(range(50)))
    gt_graph.add_edges_from(list(zip(range(10), range(1, 11))))
    edge_error_counts = {"fp": 5, "fn": 2, "ws": 1}
    vertex_error_counts = {"fp": 12, "fn": 4, "ns": 7}

    tra_error = get_ctc_tra_error(gt_graph, vertex_error_counts, edge_error_counts)

    vertex_weight_ns = 5
    vertex_weight_fn = 10
    vertex_weight_fp = 1

    edge_weight_fp = 1
    edge_weight_fn = 1.5
    edge_weight_ws = 1
    aogm0 = 50 * vertex_weight_fn + 10 * edge_weight_fn
    aogm = get_weighted_error_sum(
        vertex_error_counts,
        edge_error_counts,
        vertex_weight_ns,
        vertex_weight_fp,
        vertex_weight_fn,
        edge_weight_fp,
        edge_weight_fn,
        edge_weight_ws,
    )
    tra = 1 - min(aogm, aogm0) / aogm0

    assert pytest.approx(tra_error) == tra
