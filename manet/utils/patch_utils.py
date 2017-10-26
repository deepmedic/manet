# encoding: utf-8
from __future__ import division
import numpy as np
from manet.utils import prob_round, read_dcm
from manet.utils import cast_numpy
from manet.utils.bbox_utils import _split_bbox, _combine_bbox
from manet.utils.mask_utils import random_mask_idx, bounding_box

def extract_patch(image, bbox, pad_value=0):
    """Extract bbox from images, coordinates can be negative.

    Parameters
    ----------
    image : ndarray
       nD array
    bbox : list or tuple
       bbox of the form (coordinates, size),
       for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with height 2 and width 1.
    pad_value : number
       if bounding box would be out of the image, this is value the patch will be padded with.

    Returns
    -------
    ndarray
    """
    # Coordinates, size
    bbox_coords, bbox_size = _split_bbox(bbox)

    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0
    r_offset = (bbox_coords + bbox_size) - np.array(image.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j
                  in zip(bbox_coords + l_offset,
                         bbox_coords + bbox_size - r_offset)]
    out = image[region_idx]

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    patch = pad_value*np.ones(bbox_size, dtype=image.dtype)
    patch_idx = [slice(i, j) for i, j
                 in zip(l_offset, bbox_size + l_offset - r_offset)]
    patch[patch_idx] = out
    return patch


def rebuild_bbox(bbox, new_size):
    """Given a bounding box and a requested size return the new bounding box around the center of the old.
    If the coordinate would be non-integer, the value is randomly rounded up or down.

    Parameters
    ----------
    bbox : tuple, list or ndarray
    new_size : tuple or list


    Returns
    -------
    New bounding box.
    """
    new_size = cast_numpy(new_size)

    bbox_coords, bbox_size = _split_bbox(bbox)
    bbox_center = bbox_coords - bbox_size / 2.
    new_bbox_coords = prob_round(bbox_center - new_size / 2.)

    new_bbox = _combine_bbox(new_bbox_coords, new_size)
    return new_bbox


def sym_bbox_from_bbox(point, bbox):
    """Given a a bounding box and a point,
    the smallest box containing the bounding box around that
    point is returned."""
    bbox_coords, bbox_size = _split_bbox(bbox)

    # Compute the maximal distance between the center of mass and the bbox.
    max_dist = np.max([
        (point - bbox_coords).max(),
        (bbox_coords + bbox_size - point).max()
    ])

    new_size = (2*max_dist + 1)*np.ones(bbox_size, dtype=int)
    new_bbox_coords = point - max_dist*np.ones(bbox_size, dtype=int)
    new_bbox = _combine_bbox(new_bbox_coords, new_size)
    return new_bbox


def sym_bbox_from_point(point, bbox_size):
    """Given a size and a point, the symmetric bounding box around that point is returned.
    If there is ambiguity due to floats, the result is randomly rounded."""
    bbox_size = cast_numpy(bbox_size)
    point = cast_numpy(point)
    bbox_coords = prob_round(point - bbox_size / 2.)
    bbox = _combine_bbox(bbox_coords, bbox_size)
    return bbox


def sample_from_mask(mask, avoid, num_tries=100):
    """A random index is sampled for a mask in the non-zero values.
    As a first try, num_tries iterations randomly select a point and if found,
    proceeds. This is more efficient than finding all possible non-zero
    values which is O(n x m). If this fails within num_tries iterators, we look
    through all non-positive indices. Otherwise, we look through all
    possible indexes.

    Parameters
    ----------
    mask : str or ndarray
        Path to file containing mask or ndarray.

    Returns
    -------
    An index sampled within the mask.
    """
    if isinstance(mask, basestring):
        mask, _ = read_dcm(mask, window_leveling=False, dtype=int)

    bbox = bounding_box(mask)
    mask = extract_patch(mask, bbox)

    i = 0
    rand_idx = None
    while i < num_tries:
        # We sample up to a part of the edge
        rand_idx = tuple(
            [np.random.randint(x, y) for x, y
             in zip(avoid, mask.shape - avoid)])
        if mask[rand_idx] != 0:
            break
        i += 1
    # If that didn't work, we unfortunately have to do a full search.
    # Here we do not try to avoid the edge.
    if not rand_idx:
        rand_idx = random_mask_idx(mask)

    bbox_coords, _ = _split_bbox(bbox)
    rand_idx = cast_numpy(bbox_coords)
    idx = tuple(rand_idx + bbox_coords)
    return idx
