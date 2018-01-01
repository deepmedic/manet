# encoding: utf-8
"""Example to show how to extract candidate locations per threshold from a probability map"""
import manet.utils
from manet.feature.peak import peak_local_max
from manet.data import prob_map
import matplotlib.pyplot as plt


def get_peaks(pred, distance, threshold):
    coordinates = peak_local_max(pred, min_distance=distance, threshold=threshold)
    return coordinates


def main():
    fig = plt.figure(figsize=(16, 16))
    pred, metadata = prob_map('one')
    # spacing is in millimeters, we want at least 1.5cm in between.
    distance = 37
    coordinates = get_peaks(pred, distance, 0.5)
    plt.imshow(pred, cmap='gray')
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.', markersize=15, alpha=1)
    plt.show()


if __name__ == '__main__':
    main()
