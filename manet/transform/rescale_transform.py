# encoding: utf-8
import numpy as np
from manet._shared.utils import assert_nD
from skimage import __version__
from distutils.version import LooseVersion
from skimage.transform import rescale

SKIMAGE_VERSION = '0.14dev'


def random_rescale_2d(arr, zoom_perc):
    """Rescale an array in 2D.

    Parameters
    ----------
    arr : ndarray
        array to rescale
    zoom_percentage : float
        number between 0 and 1 to denote the percentage to scale

    Returns
    -------
    Randomly rescaled array of zoom_perc% and the selected zoom.
    """
    assert_nD(arr, 2)
    if zoom_perc < 0 or zoom_perc > 1:
        raise ValueError('zoom percentage should be in [0, 1]')

    zoom_range = 1 + zoom_perc
    if LooseVersion(__version__) < SKIMAGE_VERSION:
        raise RuntimeError('scikit-image >= %s needed for rescaling.' % SKIMAGE_VERSION)

    # scale to [2 - range, range]
    zoom = 2 - zoom_range + 2*(zoom_range - 1)*np.random.rand()
    arr = rescale(arr, zoom, anti_aliasing=True if zoom < 1 else False,
                  mode='constant', multichannel=False)
    return arr, zoom
