# encoding: utf-8
from __future__ import print_function
from __future__ import division

from manet._shared.utils import assert_nD, assert_binary, assert_prob
from manet.utils import read_image
from manet.feature.peak import peak_local_max
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.morphology import disk, closing
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox
import matplotlib.patches as mpatches
import numpy.ma as ma


def plot_2d(image, height=16, dpi=None, mask=None, bboxes=None,
            overlay=None, linewidth=2, mask_color='r', bbox_color='b',
            overlay_cmap='jet', overlay_threshold=0.1, overlay_alpha=0.1,
            overlay_local_max_min_distance=75, overlay_local_max_color='r',
            overlay_contour_color='g', save_as=None):
    """Plot image with contours.

    Parameters
    ----------
    image : ndarray
        2D image.
    height : float
        height in inches.
    dpi : int
        dpi when saving image.
    mask : ndarray
        binary mask to plot overlay.
    linewidth : float
        thickness of the overlay lines.
    mask_color : str
        matplotlib supported color for mask overlay.
    bbox_color : str
        matplotlib supported color for bbox overlay.
    overlay_cmap : str
        matplotlib support overlay cmap.
    overlay_threshold : str
        Threshold value before an overlay value is shown.
    overlay_alpha : float
        alpha value for overlay.
    save_as : str
        path where image will be saved.

    BUG: Sometimes the output has a white edge to the left of the image.
    """
    aspect = float(image.shape[1]) / image.shape[0]
    fig, ax = plt.subplots(1, figsize=(aspect*height, height))
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

    if overlay is not None:
        add_2d_overlay(overlay, ax, linewidth, threshold=overlay_threshold, cmap=overlay_cmap,
                       alpha=overlay_alpha, contour_color=overlay_contour_color)

    if overlay is not None and overlay_local_max_min_distance:
        add_local_maxima(overlay, ax, overlay_local_max_min_distance, overlay_threshold, overlay_local_max_color)

    if not save_as:
        plt.show()
    else:
        fig.gca().set_axis_off()
        fig.gca().xaxis.set_major_locator(NullLocator())
        fig.gca().yaxis.set_major_locator(NullLocator())
        fig.savefig(save_as, bbox_inches=Bbox([[0, 0], [aspect*height, height]]), pad_inches=0, dpi=dpi)
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
        2D binary array.
    axis : axis object
        matplotlib axis object.
    linewidth : float
        thickness of the overlay lines.
    color : str
        matplotlib supported color string for contour overlay.

    TODO: In utils.mask_utils we have function which computes one contour, perhaps these can be merged.
    """
    assert_nD(mask, 2, 'mask')
    assert_binary(mask, 'mask')
    contours = find_contours(mask, 0.5)

    for contour in contours:
        axes.plot(*(contour[:, [1, 0]].T), color=color, linewidth=linewidth)


def add_2d_overlay(overlay, ax, linewidth, threshold=0.1, cmap='jet', alpha=0.1, closing_radius=15, contour_color='g'):
    """Adds an overlay of the probability map and predicted regions

    overlay : ndarray
    ax : axis object
       matplotlib axis object
    linewidth : float
       thickness of the overlay lines;
    threshold : float
    cmap : str
       matplotlib supported cmap.
    alpha : float
       alpha values for the overlay.
    closing_radius : int
       radius for the postprocessing closing
    contour_color : str
       matplotlib supported color for the contour.

    """
    assert_nD(overlay, 2, 'overlay')
    assert_prob(overlay, 'overlay')
    if threshold:
        overlay = ma.masked_where(overlay < threshold, overlay)

    ax.imshow(overlay, cmap=cmap, alpha=alpha)

    if contour_color:
        mask = closing(overlay.copy(), disk(closing_radius))
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        add_2d_contours(mask, ax, linewidth=linewidth, color=contour_color)


def add_local_maxima(overlay, ax, min_distance, threshold, overlay_color='r'):
    coordinates = peak_local_max(overlay, min_distance=min_distance, threshold=threshold)
    ax.plot(coordinates[:, 1], coordinates[:, 0], overlay_color + '.', markersize=15, alpha=1)
