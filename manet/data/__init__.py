# encoding: utf-8
from manet.utils import read_image
import os


def curr_path(fn):
    dirname = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dirname, fn)


def prob_map():
    prob, metadata = read_image(curr_path('prediction.nrrd'))
    prob = prob[0]
    metadata['spacing'] = (0.2, 0.2)

    return prob, metadata
