# encoding: utf-8
"""Example to show how to extract candidate locations per threshold from a probability map"""
import manet.utils
from manet.feature.peak import peak_local_max
import matplotlib.pyplot as plt


def get_peaks(pred, threshold):
    coordinates = peak_local_max(pred, min_distance=1500/50, threshold=threshold)
    return coordinates


def main():
    fig = plt.figure(figsize=(16, 16))
    pred, metadata = manet.utils.read_image('prediction.nrrd')
    print(metadata['spacing'])


    coordinates = get_peaks(pred[0], 0.8)
    plt.imshow(pred[0], cmap='gray')
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.', markersize=15, alpha=1)
    plt.show()


if __name__ == '__main__':
    main()
