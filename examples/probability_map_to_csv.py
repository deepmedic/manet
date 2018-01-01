#!/usr/bin/env python
# encoding: utf-8
import manet.utils
from manet.feature.peak import peak_local_max
from tqdm import tqdm
import numpy as np
import os
import click


def prediction_to_csv(pred_fn, csv_fn, num_steps):
    """Given filename to probability map and number of steps create a csv file with the local maxima for that threshold"""
    pred, metadata = manet.utils.read_image(pred_fn)
    pred = pred[0]
    distance = 37  # 2 * distance + 1 = 15mm = 75 pixels (200 micron spacing)
    thresholds = np.linspace(0.1, 1, num_steps)

    f = open(csv_fn, 'a')
    f.write('thr,row,col\n')
    tqdm.write('Working on {}'.format(pred_fn))
    for threshold in tqdm(thresholds):
        coordinates = peak_local_max(pred, min_distance=distance, threshold=threshold)
        for coord in coordinates:
            coord = coord.tolist()
            f.write('{},{},{}\n'.format(threshold, coord[0], coord[1]))
    f.close()



@click.command()
@click.argument('folder_id', type=click.Path(exists=True))
def generate_csv(folder_id):
    """Given folder, folder/prediction.nrrd is given and folder/prediction.csv written."""
    pred_fn = os.path.join(folder_id, 'prediction.nrrd')
    csv_fn = os.path.join(folder_id, 'prediction.csv')

    if os.path.exists(csv_fn):
        os.remove(csv_fn)

    prediction_to_csv(pred_fn, csv_fn, 90)


if __name__ == '__main__':
    generate_csv()
