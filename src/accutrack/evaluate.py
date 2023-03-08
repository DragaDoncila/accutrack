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

def make_network_x_graph(coords: 'pd.DataFrame', edges: 'List[Tuple[int, int, int]]'):
    g = nx.DiGraph()
    kwarg_dict = coords.to_dict(orient='index')
    g.add_nodes_from(kwarg_dict.items())
    edge_tuples = [(e[0], e[1], {'is_parent': e[2]}) for e in edges]
    g.add_edges_from(edge_tuples)
    return g

def get_node_position(node_attrs):
    if 'z' in node_attrs:
        return (node_attrs['t'], node_attrs['z'], node_attrs['y'], node_attrs['x'])
    return (node_attrs['t'], node_attrs['y'], node_attrs['x'])


def get_frame_det_test_matrix(gt_graph: 'networkx.Graph', gt_frame: 'np.ndarray', gt_nodes: 'List[int]', comp_graph: 'networkx.Graph', comp_frame: 'np.ndarray', comp_nodes: 'List[int]'):
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
        gt_label = gt_attrs['label']
        gt_blob_mask = gt_frame == gt_label
        for i, comp_node_id in enumerate(comp_nodes):
            comp_attrs = comp_graph.nodes[comp_node_id]
            comp_label = comp_attrs['label']
            comp_blob_mask = comp_frame == comp_label
            is_match = int(detection_test(gt_blob_mask, comp_blob_mask))
            det_test_matrix[i, j] = is_match
            # once we've found a match for a gt marker, we can exit early 
            if is_match:
                break

    return det_test_matrix

def detection_test(gt_blob: 'np.ndarray', comp_blob: 'np.ndarray') -> bool:
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

def get_detection_matrices(gt_graph: 'nx.Graph', gt_ims: 'np.ndarray', comp_graph: 'nx.Graph', comp_ims: 'np.ndarray'):
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
    List[np.ndarray]
        Ordered list where each element is the detection matrix for frame i.
    """
    # the node IDs are contiguous, so we're going to rely on this for our matrix setup
    det_matrices = []
    for t in tqdm(range(len(comp_ims)), 'Detection tests'):
        comp_nodes = [node_id for node_id in comp_graph.nodes if comp_graph.nodes[node_id]['t'] == t]
        gt_nodes = [node_id for node_id in gt_graph.nodes if gt_graph.nodes[node_id]['t'] == t]
        comp_frame = comp_ims[t]
        gt_frame = gt_ims[t]
        det_test_matrix = get_frame_det_test_matrix(gt_graph, gt_frame, gt_nodes, comp_graph, comp_frame, comp_nodes)
        det_matrices.append(det_test_matrix)
    return det_matrices


if __name__ == '__main__':
    comp_ims, comp_coords, comp_coord_cols, comp_edges = get_comp_graph('/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/', '01')
    comp_g = make_network_x_graph(comp_coords, comp_edges)
    gt_ims, coords, coord_cols, edges = get_gt_graph('/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/', '01')
    gt_g = make_network_x_graph(coords, edges)
    det_matrices = get_detection_matrices(gt_g, gt_ims, comp_g, comp_ims)
    print("Done")
