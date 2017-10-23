# encoding: utf-8
from manet.utils import cast_numpy


def _split_bbox(bbox):
    """Split bbox into coordinates and size

    Parameters
    ----------
    bbox : tuple or ndarray. Given dimension n, first n coordinates are the starting point, the other n the size.

    Returns
    -------
    coordinates and size, both ndarrays.
    """
    bbox = cast_numpy(bbox)

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
    bbox_coords = cast_numpy(bbox_coords).astype(int)
    bbox_size = cast_numpy(bbox_size).astype(int)
    bbox = tuple(bbox_coords.tolist() + bbox_size.tolist())
    return bbox
