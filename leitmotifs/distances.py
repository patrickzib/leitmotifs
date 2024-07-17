# -*- coding: utf-8 -*-
"""Distances used in LAMA
"""

__author__ = ["patrickzib"]

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True, inline="always")
def sliding_csum(ts, m):
    """
    Computes the sliding cumulative sum of squares of a time series with a specified window size.

    Parameters:
    -----------
    ts : array-like
        The time series
    m : int
        The length of the sliding window to compute std and mean over.

    Returns:
    --------
    csumsq_diff: numpy.ndarray
        A 1-dimensional numpy array containing the difference between the sliding cumulative sum of
        squares of the time series with the current window and that with the previous window.

    """
    csumsq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts ** 2)))
    return csumsq[m:] - csumsq[:-m]


@njit(fastmath=True, cache=True, inline="always")
def euclidean_distance(dot_rolled, n, m, csumsq, order, halve_m):
    dist = -2 * dot_rolled + csumsq + csumsq[order]

    # self-join: exclusion zone
    trivialMatchRange = (max(0, order - halve_m), min(order + halve_m, n))
    dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    # allow subsequenceg itself to be in result
    dist[order] = 0

    return dist


@njit(fastmath=True, cache=True, inline="always")
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
        movmean : array-like
            The n-m+1 mean values
        movstd : array-like
            The n-m+1 std values
    """
    # if isinstance(ts, pd.Series):
    #     ts = ts.to_numpy()
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts ** 2)))
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]

    movmean = segSum / m

    # avoid dividing by too small std, like 0
    movstd = np.sqrt(np.clip(segSumSq / m - (segSum / m) ** 2, 0, None))
    movstd = np.where(np.abs(movstd) < 0.1, 1, movstd)

    return [movmean, movstd]


@njit(fastmath=True, cache=True, inline="always")
def znormed_euclidean_distance(dot_rolled, n, m, preprocessing, order, halve_m):
    """ Implementation of z-normalized Euclidean distance """
    means, stds = preprocessing
    dist = 2 * m * (1 - (dot_rolled - m * means * means[order]) / (
            m * stds * stds[order]))

    # self-join: exclusion zone
    trivialMatchRange = (max(0, order - halve_m), min(order + halve_m, n))
    dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    # allow subsequenceg itself to be in result
    dist[order] = 0

    return dist


_DISTANCE_MAPPING = {
    "znormed_euclidean": (sliding_mean_std, znormed_euclidean_distance),
    "znormed_ed": (sliding_mean_std, znormed_euclidean_distance),
    "ed": (sliding_csum, euclidean_distance),
    "euclidean": (sliding_csum, euclidean_distance),
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
