# encoding: utf-8
from __future__ import division
import numpy as np
from manet.utils import prob_round


def _split_bbox(bbox):
    """Split bbox into coordinates and size

    Parameters
    ----------
    bbox : tuple or ndarray. Given dimension n, first n coordinates are the starting point, the other n the size.

    Returns
    -------
    coordinates and size, both ndarrays.
    """
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)

    ndim = int(len(bbox) / 2)
    bbox_coords = bbox[:ndim]
    bbox_size = bbox[ndim:]
    return bbox_coords, bbox_size


def _combine_bbox(bbox_coords, bbox_size):
    """Combine coordinates and size into a bounding box.

    Parameters:
    bbox_coords : tuple or ndarray
    bbox_size : tuple or ndarray

    Returns
    -------
    bounding box

    """
    if not isinstance(bbox_coords, np.ndarray):
        bbox_coords = np.array(bbox_coords)
    if not isinstance(bbox_size, np.ndarray):
        bbox_size = np.array(bbox_size)

    bbox = tuple(bbox_coords.tolist() + bbox_size.tolist())
    return bbox


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


def resize_bbox(bbox, new_size):
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
    if not isinstance(new_size, np.ndarray):
        new_size = np.array(new_size)

    bbox_coords, bbox_size = _split_bbox(bbox)
    bbox_center = bbox_coords - bbox_size / 2.
    new_bbox_coords = prob_round(bbox_center - new_size / 2.)

    new_bbox = _combine_bbox(new_bbox_coords, new_size)
    return new_bbox


def symmetric_com_bbox(bbox, center_of_mass):
    """Given a sample with a bounding box and a center of mass,
    the smallest box containing the bounding box around the center
    of mass is returned."""
    bbox_coords, bbox_size = _split_bbox(bbox)

    # Compute the maximal distance between the center of mass and the bbox.
    max_dist = np.max([
        (center_of_mass - bbox_coords).max(),
        (bbox_coords + bbox_size - center_of_mass).max()
    ])

    new_size = (2*max_dist + 1)*np.ones(bbox_size, dtype=int)
    new_bbox_coords = center_of_mass - max_dist*np.ones(bbox_size, dtype=int)
    new_bbox = _combine_bbox(new_bbox_coords, new_size)
    return new_bbox
