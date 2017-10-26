# encoding: utf-8
import numpy as np
from manet._shared.utils import assert_nD
from skimage.transform import rotate


def random_rotate_2d(arr, angle_range):
    """Random rotate an array.

    Parameters
    ----------
    arr : ndarray
        array to zoom
    angle_range : int
        rotation range in degrees

    Returns
    -------
    Randomly rotated array between -angle_range and angle_range, and the randomly selected angle.

    """
    assert_nD(arr, 2)
    angle = np.random.randint(-angle_range, angle_range)
    arr = rotate(arr, angle, mode='constant')
    return arr, angle
