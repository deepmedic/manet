# encoding: utf-8
from __future__ import division
import numpy as np
import scipy.spatial.distance
from manet._shared.utils import assert_nD


def tpr(pred, gt, dist=1, mode='region'):
    """Compute the true positive rate of predictions ground truth (binary map). Both are lists of coordinates. If there is a point in the
    predictions list which is at most `dist` away from a point in the ground truth, this is counted as a true positive.

    If there are multiple ground truths, the result is the number of correct hits in the `region` mode, if the mode is `case` then the
    result is either 0 or 1, depending on if a region is matched or not.

    Parameters
    ----------
    pred : list or ndarray
        either a list of lists or a n x 2 ndarray.
    gt : list or ndarray
        either a list of lists or a n x 2 ndarray.
    dist : number
        distance between points which are considered as positives or negatives.
    mode : str
        either `region` or `case`.

    Returns
    -------
    float
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if mode not in ['region', 'case']:
        raise ValueError('{} not a valid mode.'.format(mode))

    assert_nD(pred, 2, 'predictions')
    assert_nD(gt, 2, 'ground truth')
    assert pred.shape[1] == gt.shape[1], 'Both the predictions and ground truth should have the same dimensions.'

    if pred.shape[1] != 2:
        raise NotImplementedError(
            'Currently only 2D comparisons are available. If you wish to extend, '
            'take into account that the distance in the third dimension is often scaled because of a different resolution.')

    hits = np.any(scipy.spatial.distance.cdist(pred, gt) <= dist, axis=0).astype(np.float)
    if mode == 'region':
        tpr = hits.sum() / hits.size
    elif mode == 'case':
        tpr = float(hits.sum() > 0)

    return tpr
