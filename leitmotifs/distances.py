# -*- coding: utf-8 -*-
"""Distances used in motiflets.
"""

__author__ = ["patrickzib"]

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True, nogil=True)
def sliding_csum(ts, m):
    """
    Computes the sliding cumulative sum of squares of a time series with a
    specified window size.

    Parameters:
    -----------
    ts : array-like
        The time series
    m : int
        The length of the sliding window to compute std and mean over.

    Returns:
    --------
    csumsq: numpy.ndarray
        A 1-dimensional numpy array containing the sliding cumulative sum of
        squares of the time series with the current window and that
        with the previous window.

    """
    csumsq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts ** 2)))
    return csumsq[m:] - csumsq[:-m]


@njit(cache=True, nogil=True)
def euclidean_distance(dot_rolled, n, m, csumsq, order, halve_m):
    dist = -2 * dot_rolled + csumsq + csumsq[order]

    # self-join: exclusion zone
    start, end = (max(0, order - halve_m), min(order + halve_m, n))
    dist[start:end] = np.inf

    # allow subsequence itself to be in result
    dist[order] = 0
    return dist


@njit(fastmath=True, cache=True, nogil=True)
def sliding_csum_dcsum(ts, m):
    """
    Computes the sliding cumulative sum of squares of a time series with a
    specified window size.

    Parameters:
    -----------
    ts : array-like
        The time series
    m : int
        The length of the sliding window to compute std and mean over.

    Returns:
    --------
    csumsq: numpy.ndarray
        A 1-dimensional numpy array containing the sliding cumulative sum of
        squares of the time series with the current window and that
        with the previous window.

    """
    csum = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts ** 2)))

    dcsum = np.concatenate((np.zeros(1, dtype=np.float64),
                            np.cumsum((ts[:-1] - ts[1:]) ** 2)))

    return csum[m:] - csum[:-m], np.sqrt(dcsum[m - 1:] - dcsum[:-(m - 1)])


@njit(cache=True, nogil=True)
def complexity_invariant_distance(dot_rolled, n, m, preprocessing, order, halve_m):
    """ Implementation of the complexity invariant distance (CID) """
    csumsq, ce = preprocessing

    ed = -2 * dot_rolled + csumsq + csumsq[order]
    ed = np.maximum(ed, 0.0)
    max_ce = np.maximum(ce, ce[order])
    min_ce = np.maximum(np.minimum(ce, ce[order]), 1e-12)
    cf = np.maximum(max_ce / min_ce, 1.0)
    dist = ed * cf

    # self-join: exclusion zone
    start, end = (max(0, order - halve_m), min(order + halve_m, n))
    dist[start:end] = np.inf

    # allow subsequence itself to be in result
    dist[order] = 0
    return dist


@njit(cache=True, nogil=True)
def cosine_distance(dot_rolled, n, m, csumsq, order, halve_m):
    denominator = np.sqrt(csumsq * csumsq[order])
    dist = np.ones(n, dtype=np.float64)
    non_zero = denominator > 0
    dist[non_zero] = 1 - dot_rolled[non_zero] / denominator[non_zero]

    # self-join: exclusion zone
    start, end = (max(0, order - halve_m), min(order + halve_m, n))
    dist[start:end] = np.inf

    # allow subsequence itself to be in result
    dist[order] = 0
    return dist


@njit(fastmath=True, cache=True, nogil=True)
def sliding_mean_std(ts, m):
    """Computes the incremental mean, std, given a time series and windows of length m.

    Computes a total of n-m+1 sliding mean and std-values.

    This implementation is efficient and in O(n), given TS length n.

    Parameters
    ----------
    ts : array-like
        The time series
    m : int
        The length of the sliding window to compute std and mean over.

    Returns
    -------
    Tuple
        moving_mean : array-like
            The n-m+1 mean values
        moving_std : array-like
            The n-m+1 std values
    """
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts ** 2)))
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]

    moving_mean = segSum / m

    # avoid dividing by too small std, like 0
    moving_std = np.sqrt(np.clip(segSumSq / m - (segSum / m) ** 2, 0, None))
    moving_std = np.where(moving_std < 1e-4, 0.1, moving_std)

    return [moving_mean, moving_std]


@njit(cache=True, nogil=True)
def znormed_euclidean_distance(dot_rolled, n, m, preprocessing, order, halve_m):
    """ Implementation of z-normalized Euclidean distance """
    means, stds = preprocessing
    dist = 2 * m * (1 - (dot_rolled - m * means * means[order]) / (
            m * stds * stds[order]))
    dist = np.maximum(dist, 0.0)

    # self-join: exclusion zone
    start, end = (max(0, order - halve_m), min(order + halve_m, n))
    dist[start:end] = np.inf

    # allow subsequence itself to be in result
    dist[order] = 0
    return dist


@njit(fastmath=True, cache=True, nogil=True, inline='always')
def znormed_euclidean_distance_single(a, b, a_i, b_j, preprocessing):
    """ Implementation of z-normalized Euclidean distance """
    means, stds = preprocessing
    m = len(a)
    dot = np.float64(0.0)
    for i in range(m):
        dot += a[i] * b[i]
    return 2 * m * (1 - (dot - m * means[a_i] * means[b_j]) / (
            m * stds[a_i] * stds[b_j]))


@njit(fastmath=True, cache=True, nogil=True, inline='always')
def euclidean_distance_single(a, b, *args):
    """ Implementation of the Euclidean distance """
    dist = np.float64(0.0)
    for i in range(len(a)):
        diff = a[i] - b[i]
        dist += diff * diff
    return dist


@njit(fastmath=True, cache=True, nogil=True, inline='always')
def cosine_distance_single(a, b, a_i, b_j, preprocessing):
    dot = np.float64(0.0)
    for i in range(len(a)):
        dot += a[i] * b[i]
    denominator = np.sqrt(preprocessing[a_i] * preprocessing[b_j])
    if denominator == 0:
        return np.float64(1.0)

    dist = 1 - dot / denominator
    return dist


@njit(fastmath=True, cache=True, nogil=True, inline='always')
def complexity_invariant_distance_single(a, b, a_i, b_j, preprocessing):
    """ Implementation of the Complexity Invariant Distance (CID) """
    _, ce = preprocessing

    ed = np.float64(0.0)
    for i in range(len(a)):
        diff = a[i] - b[i]
        ed += diff * diff

    max_ce = max(ce[a_i], ce[b_j])
    min_ce = max(min(ce[a_i], ce[b_j]), 1e-12)
    cf = max(max_ce / min_ce, 1.0)
    dist = ed * cf

    return dist


_DISTANCE_MAPPING = {
    # z-normed Euclidean Distance
    "znormed_euclidean": (
        sliding_mean_std,
        znormed_euclidean_distance, znormed_euclidean_distance_single),
    "znormed_ed": (
        sliding_mean_std,
        znormed_euclidean_distance, znormed_euclidean_distance_single),

    # Euclidean Distance
    "ed": (
        sliding_csum,
        euclidean_distance, euclidean_distance_single),
    "euclidean": (
        sliding_csum,
        euclidean_distance, euclidean_distance_single),

    # Cosine Distance
    "cosine": (
        sliding_csum,
        cosine_distance, cosine_distance_single),

    # Complexity Invariant Distance
    "CID": (
        sliding_csum_dcsum,
        complexity_invariant_distance, complexity_invariant_distance_single),
    "cid": (
        sliding_csum_dcsum,
        complexity_invariant_distance, complexity_invariant_distance_single)
}


def map_distances(distance_name):
    """
    Computes and returns the distance function and its corresponding preprocessing function, given a distance name.

    Parameters:
    -----------
    distance_name: str
        The name of the distance function to be computed. Available options are "znormed_euclidean_distance"
        and "euclidean_distance".

    Returns:
    --------
    tuple:
        A tuple containing two functions - the preprocessing function and the distance function.
        The preprocessing function takes in a time series and the window size. The distance function takes in
        the index of the subsequence, the dot product between the subsequence and all other subsequences,
        the window size, the preprocessing output, and a boolean flag indicating whether to compute the
        squared distance. It returns the distance between the two subsequences.

    Raises:
    -------
    ValueError:
        If `distance_name` is not a valid distance function name. Valid options are "znormed_euclidean_distance"
        and "euclidean_distance".
    """
    if distance_name not in _DISTANCE_MAPPING:
        raise ValueError(
            f"{distance_name} is not a valid distance. Implementations include: {', '.join(_DISTANCE_MAPPING.keys())}")

    return _DISTANCE_MAPPING[distance_name]
