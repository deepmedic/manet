# encoding: utf-8
from __future__ import print_function
from __future__ import division

from manet._shared.utils import assert_nD, assert_binary
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox
import matplotlib.patches as mpatches


def plot_2d(image, width=16, dpi=None, mask=None, bboxes=None, linewidth=2, mask_color='r', bbox_color='b', save_as=None):
    """Plot image with contours.

    Parameters
    ----------
    image : ndarray
        2D image
    width : float
        width in inches
    dpi : int
        dpi when saving image
    mask : ndarray
        binary mask to plot overlay
    linewidth : float
        thickness of the overlay lines
    mask_color : str
        matplotlib supported color for mask overlay
    bbox_color : str
        matplotlib supported color for bbox overlay
    save_as : str
        path where image will be saved
    """
    aspect = float(image.shape[0]) / image.shape[1]
    fig, ax = plt.subplots(1, figsize=(width, aspect*width))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    cmap = None
    if image.ndim == 2:
        cmap = 'gray'
    elif (image.ndim == 3 and image.shape[-1] == 1):
        image = image[..., 0]
        cmap = 'gray'

    ax.imshow(image, cmap=cmap, aspect='equal', extent=(0, image.shape[1], image.shape[0], 0))
    if mask is not None:
        add_2d_contours(mask, ax, linewidth, mask_color)

    if bboxes is not None:
        for bbox in bboxes:
            add_2d_bbox(bbox, ax, linewidth, bbox_color)

    if not save_as:
        plt.show()
    else:
        fig.gca().set_axis_off()
        fig.gca().xaxis.set_major_locator(NullLocator())
        fig.gca().yaxis.set_major_locator(NullLocator())
        fig.savefig(save_as, bbox_inches=Bbox([[0, 0], [width, aspect*width]]), pad_inches=0, dpi=dpi)
        plt.close()


def add_2d_bbox(bbox, ax, linewidth=0.5, color='b'):
    """Add bounding box to the image.

    Parameters
    ----------
    bbox : tuple
        Tuple of the form (row, col, height, width).
    axis : axis object
    linewidth : float
        thickness of the overlay lines
    color : str
        matplotlib supported color string for contour overlay.

    """
    rect = mpatches.Rectangle(bbox[:2][::-1], bbox[3], bbox[2],
                              fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)


def add_2d_contours(mask, axes, linewidth=0.5, color='r'):
    """Plot the contours around the `1`'s in the mask

    Parameters
    ----------
    mask : ndarray
        2D binary array
    axis : axis object
    linewidth : float
        thickness of the overlay lines
    color : str
        matplotlib supported color string for contour overlay.

    TODO: In utils.mask_utils we have function which computes one contour, perhaps these can be merged.
    """
    assert_nD(mask, 2, 'mask')
    assert_binary(mask, 'mask')
    contours = find_contours(mask, 0.5)

    for contour in contours:
        axes.plot(*(contour[:, [1, 0]].T), color=color, linewidth=linewidth)
