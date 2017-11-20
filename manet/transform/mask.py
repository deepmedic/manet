# encoding: utf-8
# Author: Jonas Teuwen
import numpy as np
from skimage.transform import resize as _resize
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
    assert_nD(mask, 2)
    assert_binary(mask, 2)

    out = _resize(
        mask.astype(np.float), output_shape, order=0, mode='constant', cval=0,
        clip=True, preserve_range=False)

    return out.astype('uint8')
