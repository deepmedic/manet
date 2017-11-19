# encoding: utf-8
from __future__ import print_function
from __future__ import division

from manet._shared.utils import assert_nD, assert_binary
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.ticker import NullLocator


def plot_2d(image, height=16, dpi=600, mask=None, bboxes=None, mask_color='r', bbox_color='b', linewidth=0.5, save_as=None):
    """Plot image with contours.

    TODO: Fix annoying 1 pixel edge.
    """
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    cmap = None
    if image.ndim == 2:
        cmap = 'gray'
    elif (image.ndim == 3 and image.shape[-1] == 1):
        image = image[..., 0]
        cmap = 'gray'

    ax.imshow(image, cmap=cmap, aspect='equal')
    if mask is not None:
        add_2d_contours(mask, ax, linewidth, mask_color)

    if bboxes is not None:
        add_2d_bbox(bboxes, linewidth, bbox_color)

    if not save_as:
        plt.show()
    else:
        fig.gca().set_axis_off()
        fig.gca().xaxis.set_major_locator(NullLocator())
        fig.gca().yaxis.set_major_locator(NullLocator())
        fig.savefig(save_as, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()


def add_2d_bbox(axes, linewidth=0.5, color='b'):
    raise NotImplementedError('Not yet implemented')


def add_2d_contours(mask, axes, linewidth=0.5, color='r'):
    """Plot the contours around the `1`'s in the mask"""
    assert_nD(mask, 2, 'mask')
    assert_binary(mask, 'mask')
    contours = find_contours(mask, 0.5)
    for contour in contours:
        axes.plot(*(contour[:, [1, 0]].T), color=color, linewidth=linewidth)
