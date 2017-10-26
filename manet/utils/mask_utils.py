# encoding: utf-8
import numpy as np
from manet.utils.bbox_utils import _combine_bbox
from manet._shared.utils import assert_nD, assert_binary
from skimage.measure import find_contours, approximate_polygon


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


def find_contour(mask, tolerance=0):
    """"""

    assert_nD(mask, 2)
    assert_binary(mask)

    contours = find_contours(mask, 0.5)
    if len(contours) != 1:
        raise ValueError('To find the contour, the mask cannot have holes.')

    contour = contours[0]

    # Check if contour is closed
    if np.all(contour[0] == contour[-1]):
        return contour

    # Check if mask is to the left or right
    # Find most left point
    point_0 = contour[0]
    point_1 = contour[1]
    point_end = contour[-1]

    diff = point_1 - point_0
    is_left = diff[0] > 0

    # If mask is left, connect first point along a straight line to the edge
    if is_left:
        new_start = np.array([point_0[0], 0])
        new_end = np.array([point_end[0], 0])
    else:
        new_start = np.array([point_0[0], mask.shape[1] - 1])
        new_end = np.array([point_end[0], mask.shape[1] - 1])

    contour = np.vstack([new_start, contour, new_end, new_start])
    contour = approximate_polygon(contour, tolerance)

    return contour
