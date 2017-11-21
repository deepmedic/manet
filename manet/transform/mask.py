# encoding: utf-8
# Author: Jonas Teuwen
import numpy as np
from skimage.transform import resize as _resize
from skimage.measure import label, regionprops
from manet._shared.utils import assert_nD, assert_binary


def resize(mask, output_shape):
    """Resize a mask to the requested output shape with nearest neighbors

    Parameters
    ----------
    mask : ndarray
        binary array
    output_shape : tuple
        requested shape

    Returns
    -------
    output_shape : tuple
        requested output shape


    TODO: It might be possible to combine this function with one of the transforms
    """
    assert_nD(mask, 2, 'mask')
    assert_binary(mask, 'mask')

    out = _resize(
        mask.astype(np.float), output_shape, order=0, mode='constant', cval=0,
        clip=True, preserve_range=False)

    return out.astype('uint8')


def bounding_box(mask):
    """Compute the bounding box for a given binary mask, and the center of mass

    Parameters
    ----------
    mask : ndarray
        binary array

    Returns
    -------
    List of ((col, row, height, width), (col, row)) tuples.
    """
    assert_nD(mask, 2, 'mask')
    assert_binary(mask, 'mask')

    regions = regionprops(label(mask))
    bboxes = []
    for region in regions:
        center_of_mass = map(int, region.centroid)
        bbox_ = np.array(region.bbox).astype(np.int)
        bbox = np.concatenate([bbox_[:2], bbox_[2:] - bbox_[:2]]).tolist()
        bboxes.append((bbox, center_of_mass))

    return bboxes
