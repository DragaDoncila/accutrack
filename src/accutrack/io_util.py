import glob
import os
from typing import Tuple
import pandas as pd
from tifffile import TiffFile
import numpy as np


def peek(im_file: str) -> Tuple[Tuple[int], 'np.dtype']:
    """Return shape and dtype of TIFF file at path im_file

    Returns
    -------
    Tuple[Tuple[int], 'np.dtype']
        Shape and dtype of first image page.
    """
    with TiffFile(im_file) as im:
        im_shape = im.pages[0].shape
        im_dtype = im.pages[0].dtype
    return im_shape, im_dtype

def load_tiff_frames(im_dir: str) -> 'np.ndarray':
    """Returned stacked array of all tiffs in im_dir

    Parameters
    ----------
    im_dir : str
        path to directory containing one or more *.tif files

    Returns
    -------
    np.ndarray
        array of stacked tiffs
    """
    all_tiffs = list(sorted(glob.glob(f'{im_dir}*.tif')))
    n_frames = len(all_tiffs)
    frame_shape, im_dtype = peek(all_tiffs[0])
    im_array = np.zeros((n_frames, *frame_shape), dtype=im_dtype)
    for i, tiff_pth in enumerate(all_tiffs):
        with TiffFile(tiff_pth) as im:
            im_array[i] = im.pages[0].asarray()
    return im_array

def load_st_seg(data_dir, seq):
    seg_st_path = os.path.join(data_dir, f"{seq}_ST", "SEG/")
    seg_ims = load_tiff_frames(seg_st_path)
    return seg_ims

def load_track_info(data_dir, seq, is_gt=False):
    if is_gt:
        track_im_path = os.path.join(data_dir, f"{seq}_GT/", "TRA/")
        track_path = os.path.join(track_im_path, 'man_track.txt')
    else:
        track_im_path = os.path.join(data_dir, f"{seq}_RES/")
        track_path = os.path.join(track_im_path, 'res_track.txt')

    track_ims = load_tiff_frames(track_im_path)
    track_df = pd.read_csv(track_path, sep=' ', names=['id', 'min_frame', 'max_frame', 'parent'], header=None)
    return track_ims, track_df
