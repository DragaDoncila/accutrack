from typing import List, Tuple
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.graph import pixel_graph, central_pixel

def get_graph_from_ims(track_ims: 'np.ndarray', track_df: 'pd.DataFrame') -> Tuple['pd.DataFrame', List[Tuple[int, int, int]]]:
    """Extract vertices and edge list from track_ims and track_df

    Parameters
    ----------
    track_ims : np.ndarray
        array containing tracked annotations for cells
    track_df : pd.DataFrame
        (id, min_frame, max_frame, parent) track df

    Returns
    -------
    coords, edges
    """
    coords, coord_cols = extract_coords(track_ims)
    link_edges = get_link_edges(coords, track_df['id'].max())
    parent_edges = get_parent_edges(coords, track_df)
    return coords, coord_cols, link_edges + parent_edges


def get_link_edges(coords, max_track):
    """Construct link edges by iterating through coords rows
    of same track_id and getting source-destination vertices.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    Parameters
    ----------
    coords : pd.DataFrame
        coords dataframe of columns ('t', ['z'], 'y', 'x', 'label')
    max_track : int
        largest track ID
    """
    link_edges = []
    for track_id in range(max_track+1):
        track_rows = coords[coords.label == track_id]
        if len(track_rows):
            track_rows.sort_values(by='t')
            edge_list = list(zip(track_rows.index[:-1], track_rows.index[1:], [0 for _ in range(len(track_rows) - 1)]))
            link_edges.extend(edge_list)
    return link_edges


def get_parent_edges(coords, track_df):
    parent_edges = []
    child_rows = track_df[track_df.parent != 0]
    for index, row in child_rows.iterrows():
        src_label = row['parent']
        src_t = track_df[track_df.id == src_label]['max_frame'].values[0]
        src_index = coords[(coords.label == src_label) & (coords.t == src_t)].index.values[0]

        dest_label = row['id']
        dest_t = row['min_frame']
        dest_index = coords[(coords.label == dest_label) & (coords.t == dest_t)].index.values[0]
        
        parent_edges.append((src_index, dest_index, 1))
    return parent_edges
        

def extract_coords(ims: 'np.ndarray', is_4d: bool=False) -> 'pd.DataFrame':
    """Given images return coordinate dataframe of center
    point of each located object and its original label
    in the image

    Parameters
    ----------
    ims : np.ndarray
        segmented images to extract center coordinates
    is_4d : bool
        if data is 4d 

    Returns
    -------
    pd.DataFrame
        coords dataframe of columns ('t', ['z'], 'y', 'x', 'label')
    """
    coord_cols = ['z', 'y', 'x'] if is_4d else ['y', 'x']
    coords_df = extract_im_centers(ims, coord_cols)
    return coords_df, coord_cols


def get_real_center(prop):
    if prop.solidity > 0.9:
        return prop.centroid
    
    # shape is too convex to use centroid, get center from pixelgraph
    region = prop.image
    g, nodes = pixel_graph(region, connectivity=2)
    medoid_offset, _ = central_pixel(
            g, nodes=nodes, shape=region.shape, partition_size=100
            )
    medoid_offset = np.asarray(medoid_offset)
    top_left = np.asarray(prop.bbox[:region.ndim])
    medoid = tuple(top_left + medoid_offset)
    return medoid

def get_centers(segmentation):
    n_frames = segmentation.shape[0]
    centers_of_mass = []
    for i in range(n_frames):
        current_frame = segmentation[i]
        props = regionprops(current_frame)
        current_centers = [(i, ) + get_real_center(prop) for prop in props]
        centers_of_mass.extend(current_centers)
    return centers_of_mass


def extract_im_centers(im_arr, coord_cols):
    center_coords = np.asarray(get_centers(im_arr))
    coords_df = pd.DataFrame(center_coords, columns=['t', *coord_cols])
    coords_df['t'] = coords_df['t'].astype(int)
    labels = []
    for coord in center_coords:
        labels.append(im_arr[tuple(coord.astype(int))])
    coords_df['label'] = labels
    return coords_df

