# encoding: utf-8
"""Command-line tool to write images to file"""
import matplotlib
matplotlib.use('Agg')

import click
from manet.utils import read_dcm
from manet.transform.mask import resize, bounding_box
from manet.plotting.imshow import plot_2d


@click.command()
@click.argument('image', type=click.Path(exists=True))
@click.option('--mask', default=None,
              help='Location mask dcm', type=click.Path(exists=True))
@click.option('--overlay', default=None,
              help='Location to overlay map', type=click.Path(exists=True))
@click.option('--output', default='output.png', help='Image to write to')
@click.option('--height', default=16, help='height of the image in inches')
@click.option('--dpi', default=None, help='dpi of the output image')
@click.option('--linewidth', default=2, help='linewidth of the contours.')
@click.option('--bbox/--no-bbox', default=False, help='Plot bounding box')
@click.option('--contour/--no-contour', default=True, help='Do not plot contour.')
@click.option('--threshold', default=0.5, help='Threshold for the overlay')
@click.option('--alpha', default=0, 'alpha of the overlay.')
def write_image(image, mask, overlay, output, height, dpi, linewidth, bbox, contour, threshold, alpha):
    """Write image to disk, given input dcm. Possible to add contours and bounding boxes.
    """
    image, _ = read_dcm(image, window_leveling=True)
    if mask:
        mask_arr, _ = read_dcm(mask, window_leveling=False)
        if image.shape != mask_arr.shape:
            mask_arr = resize(mask_arr, image.shape)

    if bbox:
        bboxes = bounding_box(mask_arr)
        bboxes = [x for x, _ in bboxes]
    else:
        bboxes = None

    if not contour:
        # If we do not want to show the contour, we set it to None.
        mask_arr = None

    plot_2d(image, height=height, mask=mask_arr, bboxes=bboxes,
            overlay=overlay, overlay_cmap='jet', overlay_alpha=alpha,
            overlay_threshold=threshold, save_as=output, dpi=dpi, linewidth=linewidth)

    print('Output written to {}.'.format(output))


if __name__ == '__main__':
    write_image()
