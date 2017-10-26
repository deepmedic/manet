# encoding: utf-8
import numpy as np


def assert_nD(arr, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.
    """
    arr = np.asanyarray(arr)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if arr.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if not arr.ndim in ndim:
        raise ValueError(msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim])))


def assert_binary(arr, arg_name='image'):
    """Verify that an array is binary and non-zero.

    Parameters
    ----------
    array : array-like
    arg_name : str, optional
        The name of the array in the original function.
    """
    msg_non_binary_array = "The parameter `%s` has to be a binary-valued array."
    msg_non_zero_array = "The parameter `%s` has to be a non-zero array."
    arr = np.asanyarray(arr)
    if not np.array_equal(arr, arr.astype(bool)):
        raise ValueError(msg_non_binary_array % (arg_name))
    if arr.sum() == 0:
        raise ValueError(msg_non_zero_array % (arg_name))


