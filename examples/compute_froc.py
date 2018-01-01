# encoding: utf-8
import os
import click
from tqdm import tqdm
from manet.utils.file_readers import read_list, read_predictions_csv, write_json
from manet.metrics.froc import froc


def read_cases(cases):
    out = []
    for case in tqdm(cases):
        preds = read_predictions_csv(case)
        out.append(preds)
    return out

def read_gtrs(cases):
    out = []
    for case in tqdm(cases):
        gtrs = read_list(case)[1]
        gtrs = [float(_) for _ in gtrs.split(',')]
        out.append(gtrs)
    return out



@click.command()
@click.argument('positive_path', type=click.Path(exists=True))
@click.argument('normals_path', type=click.Path(exists=True))
@click.option('--path', default=None,
              help='Path to folders', type=click.Path(exists=True))
@click.option('--overlay', default=None,
              help='Location to overlay map', type=click.Path(exists=True))
@click.option('--output', default='output.png', help='Image to write to')
@click.option('--height', default=16, help='height of the image in inches')
@click.option('--dpi', default=None, help='dpi of the output image')
@click.option('--linewidth', default=2, help='linewidth of the contours.')
@click.option('--bbox/--no-bbox', default=False, help='Plot bounding box')
@click.option('--contour/--no-contour', default=True, help='Do not plot contour.')
@click.option('--threshold', default=0.5, help='Threshold for the overlay')
@click.option('--alpha', default=0, help='alpha of the overlay.')
def compute_froc(positive_path, normals_path, path, pos_pred, gtr, neg_pred):
    positives = read_list(positive_path)
    normals = read_list(normals_path)

    normals = [os.path.join(path, normal, neg_pred) for normal in normals]
    positives = [os.path.join(path, positive, pos_pred) for positive in positives]
    gtrs = [os.path.join(path, positive, gtr) for positive in positives]

    tqdm.write('Parsing all normals.')
    normals = read_cases(normals)
    tqdm.write('Parsing all positives.')
    positives = read_cases(positives)
    tqdm.write('Parsing all ground truths')
    gtrs = read_gtrs(gtrs)

    tqdm.write('Computing froc and saving to froc.json')
    fd = froc(positives, gtrs, normals)
    write_json(fd, 'froc.json')


if __name__ == '__main__':
    pass
