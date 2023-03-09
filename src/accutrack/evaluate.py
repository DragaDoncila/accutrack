import numpy as np
from tqdm import tqdm
from accutrack.io_util import load_track_info
from accutrack.util import get_graph_from_ims
from scipy.sparse import lil_matrix
import networkx as nx


def get_gt_graph(data_dir, seq):
    track_ims, track_df = load_track_info(data_dir, seq, is_gt=True)
    coords, coord_cols, edges = get_graph_from_ims(track_ims, track_df)
    return track_ims, coords, coord_cols, edges


def get_comp_graph(data_dir, seq):
    track_ims, track_df = load_track_info(data_dir, seq)
    coords, coord_cols, edges = get_graph_from_ims(track_ims, track_df)
    return track_ims, coords, coord_cols, edges


def make_network_x_graph(coords: "pd.DataFrame", edges: "List[Tuple[int, int, int]]"):
    g = nx.DiGraph()
    kwarg_dict = coords.to_dict(orient="index")
    g.add_nodes_from(kwarg_dict.items())
    edge_tuples = [(e[0], e[1], {"is_parent": e[2]}) for e in edges]
    g.add_edges_from(edge_tuples)
    return g


def get_node_position(node_attrs):
    if "z" in node_attrs:
        return (node_attrs["t"], node_attrs["z"], node_attrs["y"], node_attrs["x"])
    return (node_attrs["t"], node_attrs["y"], node_attrs["x"])


def get_frame_det_test_matrix(
    gt_graph: "networkx.Graph",
    gt_frame: "np.ndarray",
    gt_nodes: "List[int]",
    comp_graph: "networkx.Graph",
    comp_frame: "np.ndarray",
    comp_nodes: "List[int]",
):
    """Return detection matrix for all reference markers in the given frame.

    For each reference marker, iterate through computed markers until one is
    found that passes the detection test. If a matching computed marker is
    found, break early.

    Parameters
    ----------
    gt_graph : networkx.Graph
        Graph of ground truth tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.
    gt_frame : np.ndarray
        Ground truth image frame for which the detection test is
        being performed.
    gt_nodes : List[int]
        A list of the ground truth node IDs in the current frame.
    comp_graph : networkx.Graph
        Graph of computed tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.
    comp_frame : np.ndarray
        Computed image frame for which the detection test is
        being performed.
    comp_nodes : List[int]
        A list of the computed node IDs in the current frame.

    Returns
    -------
    np.ndarray
        boolean array of shape (len(comp_nodes), len(gt_nodes)) where
        ones indicate matching nodes
    """
    det_test_matrix = np.zeros((len(comp_nodes), len(gt_nodes)), dtype=np.uint8)
    for j, gt_node_id in enumerate(gt_nodes):
        gt_attrs = gt_graph.nodes[gt_node_id]
        gt_label = gt_attrs["label"]
        gt_blob_mask = gt_frame == gt_label
        for i, comp_node_id in enumerate(comp_nodes):
            comp_attrs = comp_graph.nodes[comp_node_id]
            comp_label = comp_attrs["label"]
            comp_blob_mask = comp_frame == comp_label
            is_match = int(detection_test(gt_blob_mask, comp_blob_mask))
            det_test_matrix[i, j] = is_match
            # once we've found a match for a gt marker, we can exit early
            if is_match:
                break

    return det_test_matrix


def detection_test(gt_blob: "np.ndarray", comp_blob: "np.ndarray") -> bool:
    """Check if computed marker overlaps majority of the reference marker.

    Given a reference marker and computer marker in original coordinates,
    return True if the computed marker overlaps strictly more than half
    of the reference marker's pixels, otherwise False.

    Parameters
    ----------
    gt_blob : np.ndarray
        2D or 3D boolean mask representing the pixels of the ground truth
        marker
    comp_blob : np.ndarray
        2D or 3D boolean mask representing the pixels of the computed
        marker

    Returns
    -------
    bool
        True if computed marker majority overlaps reference marker, else False.
    """
    n_gt_pixels = np.sum(gt_blob)
    intersection = np.logical_and(gt_blob, comp_blob)
    comp_blob_matches_gt_blob = int(np.sum(intersection) > 0.5 * n_gt_pixels)
    return comp_blob_matches_gt_blob


def get_detection_matrices(
    gt_graph: "nx.Graph",
    gt_ims: "np.ndarray",
    comp_graph: "nx.Graph",
    comp_ims: "np.ndarray",
):
    """Get detection matrices for every frame in the dataset

    Parameters
    ----------
    gt_graph : nx.Graph
        Graph of ground truth tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.
    gt_ims : np.ndarray
        Integer 2D+T or 3D+T segmentation for which the tracking solution
        has been constructed
    comp_graph : nx.Graph
        Graph of computed tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.
    comp_ims : np.ndarray
        Integer 2D+T or 3D+T segmentation for which the tracking solution
        has been constructed

    Returns
    -------
    Dict
        Dictionary indexed by t holding `det`, `comp_ids` and `gt_ids`
    """
    # the node IDs are contiguous, so we're going to rely on this for our matrix setup
    det_matrices = {}
    for t in tqdm(range(len(comp_ims)), "Detection tests"):
        comp_nodes = [
            node_id
            for node_id in comp_graph.nodes
            if comp_graph.nodes[node_id]["t"] == t
        ]
        gt_nodes = [
            node_id for node_id in gt_graph.nodes if gt_graph.nodes[node_id]["t"] == t
        ]
        comp_frame = comp_ims[t]
        gt_frame = gt_ims[t]
        det_test_matrix = get_frame_det_test_matrix(
            gt_graph, gt_frame, gt_nodes, comp_graph, comp_frame, comp_nodes
        )
        det_matrices[t] = {
            "det": det_test_matrix,
            "comp_ids": comp_nodes,
            "gt_ids": gt_nodes,
        }
    return det_matrices


def get_node_matching_map(detection_matrices: "Dict"):
    """Return list of tuples of (gt_id, comp_id) for all matched nodes

    Parameters
    ----------
    detection_matrices : Dict
        Dictionary indexed by t holding `det`, `comp_ids` and `gt_ids`

    Returns
    -------
    matched_nodes: List[Tuple[int, int]]
        List of tuples (gt_node_id, comp_node_id) denoting matched nodes
        between reference graph and computed graph
    """
    matched_nodes = []
    for m_dict in detection_matrices.values():
        matrix = m_dict["det"]
        comp_nodes = np.asarray(m_dict["comp_ids"])
        gt_nodes = np.asarray(m_dict["gt_ids"])
        row_idx, col_idx = np.nonzero(matrix)
        comp_node_ids = comp_nodes[row_idx]
        gt_node_ids = gt_nodes[col_idx]
        matched_nodes.extend(list(zip(gt_node_ids, comp_node_ids)))
    return matched_nodes


def get_vertex_errors(
    gt_graph: "networkx.Graph",
    comp_graph: "networkx.Graph",
    detection_matrices: "Dict",
):
    """Count vertex errors and assign class to each comp/gt node.

    Parameters
    ----------
    gt_graph : networkx.Graph
        Graph of ground truth tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.
    comp_graph : networkx.Graph
        Graph of computed tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.
    detection_matrices : Dict
        Dictionary indexed by t holding `det`, `comp_ids` and `gt_ids`
    """
    tp_count = 0
    fp_count = 0
    fn_count = 0
    ns_count = 0

    nx.set_node_attributes(comp_graph, False, "is_tp")
    nx.set_node_attributes(comp_graph, False, "is_fp")
    nx.set_node_attributes(comp_graph, False, "is_ns")
    nx.set_node_attributes(gt_graph, False, "is_fn")

    for t in sorted(detection_matrices.keys()):
        mtrix = detection_matrices[t]["det"]
        comp_ids = detection_matrices[t]["comp_ids"]
        gt_ids = detection_matrices[t]["gt_ids"]

        tp_rows = np.ravel(np.argwhere(np.sum(mtrix, axis=1) == 1))
        fp_rows = np.ravel(np.argwhere(np.sum(mtrix, axis=1) == 0))
        fn_cols = np.ravel(np.argwhere(np.sum(mtrix, axis=0) == 0))
        ns_rows = np.ravel(np.argwhere(np.sum(mtrix, axis=1) > 1))

        for row in tp_rows:
            node_id = comp_ids[row]
            comp_graph.nodes[node_id]["is_tp"] = True

        for row in fp_rows:
            node_id = comp_ids[row]
            comp_graph.nodes[node_id]["is_fp"] = True

        for col in fn_cols:
            node_id = gt_ids[col]
            gt_graph.nodes[node_id]["is_fn"] = True

        # num operations needed to fix a non split vertex is
        # num reference markers matched to computed marker - 1
        for row in ns_rows:
            node_id = comp_ids[row]
            comp_graph.nodes[node_id]["is_ns"] = True
            number_of_splits = np.sum(mtrix[row]) - 1
            ns_count += number_of_splits

        tp_count += len(tp_rows)
        fp_count += len(fp_rows)
        fn_count += len(fn_cols)

    error_counts = {"tp": tp_count, "fp": fp_count, "fn": fn_count, "ns": ns_count}
    return error_counts


def get_comp_subgraph(comp_graph: "networkx.Graph") -> "networkx.Graph":
    """Return computed graph subgraph of TP vertices and their incident edges.

    Parameters
    ----------
    comp_graph : networkx.Graph
        Graph of computed tracking solution. Nodes must have label
        attribute denoting the pixel value of the marker.

    Returns
    -------
    induced_graph : networkx.Graph
        Subgraph of comp_graph with only TP vertices and their incident edges
    """
    tp_nodes = [node for node in comp_graph.nodes if comp_graph.nodes[node]["is_tp"]]
    induced_graph = nx.Graph(comp_graph.subgraph(tp_nodes).copy())
    return induced_graph


def assign_edge_errors(gt_graph, comp_graph, node_mapping):
    induced_graph = get_comp_subgraph(comp_graph)

    nx.set_edge_attributes(comp_graph, False, "is_fp")
    nx.set_edge_attributes(comp_graph, False, "is_tp")
    nx.set_edge_attributes(comp_graph, False, "is_wrong_semantic")
    nx.set_edge_attributes(gt_graph, False, "is_fn")

    # fp edges - edges in induced_graph that aren't in gt_graph
    for edge in induced_graph.edges:
        source, target = edge[0], edge[1]
        source_gt_id = list(filter(lambda mp: mp[1] == source, node_mapping))[0]
        target_gt_id = list(filter(lambda mp: mp[1] == target, node_mapping))[0]
        expected_gt_edge = (source_gt_id, target_gt_id)
        if expected_gt_edge not in gt_graph.edges:
            comp_graph.edges[edge]["is_fp"] = True
        else:
            # check if semantics are correct
            is_parent_gt = gt_graph[expected_gt_edge]["is_parent"]
            is_parent_comp = comp_graph[edge]["is_parent"]
            if is_parent_gt != is_parent_comp:
                comp_graph.edges["is_wrong_semantic"] = True
            else:
                comp_graph.edges[edge]["is_tp"] = True

    # fn edges - edges in gt_graph that aren't in induced graph
    for edge in gt_graph.edges:
        source, target = edge[0], edge[1]
        source_comp_id = list(filter(lambda mp: mp[0] == source, node_mapping))[1]
        target_comp_id = list(filter(lambda mp: mp[0] == target, node_mapping))[1]
        expected_comp_edge = (source_comp_id, target_comp_id)
        if expected_comp_edge not in induced_graph.edges:
            gt_graph.edges["is_fn"] = True


def get_error_counts(gt_graph, comp_graph):
    count_fp = 0
    count_wrong_sem = 0
    count_tp = 0

    for edge in comp_graph.edges:
        if comp_graph.edges[edge]["is_fp"]:
            count_fp += 1
        elif comp_graph.edges[edge]["is_wrong_semantic"]:
            count_wrong_sem += 1
        elif comp_graph.edges[edge]["is_tp"]:
            count_tp += 1
    count_fn = len([edge for edge in gt_graph.edges if gt_graph.edges[edge]["is_fn"]])

    edge_errors = {
        "tp": count_tp,
        "fp": count_fp,
        "fn": count_fn,
        "ws": count_wrong_sem,
    }
    return edge_errors


if __name__ == "__main__":
    comp_ims, comp_coords, comp_coord_cols, comp_edges = get_comp_graph(
        "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/", "01"
    )
    comp_g = make_network_x_graph(comp_coords, comp_edges)
    gt_ims, coords, coord_cols, edges = get_gt_graph(
        "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/", "01"
    )
    gt_g = make_network_x_graph(coords, edges)
    det_matrices = get_detection_matrices(gt_g, gt_ims, comp_g, comp_ims)
    mapping = get_node_matching_map(det_matrices)
    vertex_errors = get_vertex_errors(gt_g, comp_g, det_matrices)
    assign_edge_errors(gt_g, comp_g, mapping)
    edge_errors = get_error_counts(gt_g, comp_g)
    print("Done!")
