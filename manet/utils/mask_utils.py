# encoding: utf-8
import numpy as np
from manet.utils.bbox_utils import _combine_bbox


def bounding_box(mask):
    """
    Computes the bounding box of a mask
    Parameters
    ----------
    mask : array-like
        Input mask

    Returns
    -------
    Bounding box
    """
    bbox_coords = []
    bbox_sizes = []
    for idx in range(mask.ndim):
        axis = tuple([i for i in range(mask.ndim) if i != idx])
        nonzeros = np.any(mask, axis=axis)
        min_val, max_val = np.where(nonzeros)[0][[0, -1]]
        bbox_coords.append(min_val)
        bbox_sizes.append(max_val - min_val + 1)

    return _combine_bbox(bbox_coords, bbox_sizes)


def random_mask_idx(mask):
    """Get a random mask index"""
    mask_idx = np.where(mask != 0)  # This is slow.
    try:
        selected_idx = np.random.randint(len(mask_idx[0]))
    except ValueError:
        raise ValueError('Mask contains no-zero entries.')
    rand_idx = tuple([row[selected_idx] for row in mask_idx])

    return rand_idx
