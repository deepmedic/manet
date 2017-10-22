# encoding: utf-8
import numpy as np


def prob_round(x):
    """Function to randomly round up or down.
    The part closest to the nearest integer is used as the probability to
    select either floor or ceil.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    Randomly rounded input
    """
    if not isinstance(x, np.ndarray) and not hasattr(x, '__len__'):
        x = np.array([x])

    sign = np.sign(x)
    x = np.abs(x)

    round_up = np.random.random(x.shape) < x - np.floor(x)

    x[round_up] = np.ceil(x[round_up])
    x[~round_up] = np.floor(x[~round_up])

    x = (sign * x).astype(np.int)
    # If the input is an integer, we need to output an integer.
    if x.size == 1:
        x = x[0]

    return x
