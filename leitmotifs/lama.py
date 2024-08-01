# -*- coding: utf-8 -*-
"""Compute leitmotifs using LAMA.
"""

__author__ = ["patrickzib"]

from ast import literal_eval
from os.path import exists
import os

# import numpy as np
import numpy.fft as fft
import pandas as pd
from numba import njit, prange, objmode, types
from numba.typed import Dict, List
from scipy.signal import argrelextrema
from scipy.stats import zscore
from tqdm.auto import tqdm

from leitmotifs.distances import *


def _resample(data, sampling_factor=10000):
    """Resamples a time series to roughly `sampling_factor` points.

    The method searches a factor to skip every i-th point.

    Parameters
    ----------
    data : array-like
        The time series data
    sampling_factor :
        The rough size of the time series after sampling

    Returns
    -------
    Tuple
        data :
            The raw data after sampling
        factor : int
            The factor used to sample the time series

    """
    factor = 1
    if data.shape[-1] > sampling_factor:
        factor = np.int32(data.shape[-1] / sampling_factor)
        if data.ndim >= 2:
            data = data[:, ::factor]
        else:
            data = data[::factor]
    return data, factor


def read_ground_truth(dataset):
    """Reads the ground-truth data for the time series.

    Parameters
    ----------
    dataset : String
        Name of the dataset

    Returns
    -------
    Series : pd.Series
        A series of ground-truth data

    """
    if "_gt.csv" not in dataset:
        file = os.path.splitext(dataset)[0] + "_gt.csv"
    else:
        file = os.path.splitext(dataset)[0]

    if exists(file):
        # print(file)
        series = pd.read_csv(file, index_col=0)

        for i in range(0, series.shape[0]):
            series.iloc[i] = series.iloc[i].apply(literal_eval)

        return series
    else:
        print("No ground truth found for ", dataset)
    return None

#######################
# TODO remove - only for noise experiments
noise_level = None

def add_gaussian_noise(df, noise_level=None):
    if noise_level:
        df_noisy = df.copy()
        for col in df_noisy.columns:
            std_dev = df_noisy[col].std()
            noise = np.random.normal(0, std_dev * noise_level, df_noisy[col].shape)
            df_noisy[col] += noise
        return df_noisy
    else:
        return df

#######################


def read_audio_from_dataframe(pandas_file_url, channels=None):
    """Reads a time series with an index (e.g. time) from a CSV with MFCC features."""
    global noise_level

    df = pd.read_csv(pandas_file_url, index_col=0, compression='gzip')
    audio_length_seconds = 2 * float(df.columns[-1]) - float(df.columns[-2])

    if channels:
        df = df.loc[channels]

    df_gt = read_ground_truth(pandas_file_url)

    if noise_level:         # TODO only for experiments
        print("Adding noise to the data", noise_level)
        df = add_gaussian_noise(df, noise_level)

    return audio_length_seconds, df, np.float64(df.columns), df_gt


def read_dataset_with_index(dataset, sampling_factor=10000):
    """Reads a time series with an index (e.g. time) and resamples.

    Parameters
    ----------
    dataset : String
        File location.
    sampling_factor :
        The time series is sampled down to roughly this number of points by skipping
        every other point.

    Returns
    -------
    Tuple
        data : pd.Series
            The time series (z-score applied) with the index.
        gt : pd:series
            Ground-truth, if available as `dataset`_gt file

    """
    full_path = '../datasets/ground_truth/' + dataset
    data = pd.read_csv(full_path, index_col=0).squeeze('columns')
    print("Dataset Original Length n: ", len(data))

    data, factor = _resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    data[:] = zscore(data)

    gt = read_ground_truth(dataset)
    if gt is not None:
        if factor > 1:
            for column in gt:
                gt[column] = gt[column].transform(lambda l: (np.array(l)) // factor)
        return data, gt
    else:
        return data


def pd_series_to_numpy(data):
    """Converts a PD.Series to two numpy arrays by extracting the raw data and index.

    Parameters
    ----------
    data : array or PD.Series
        the TS

    Returns
    -------
    Tuple
        data_index : array_like
            The index of the time series
        data_raw :
            The raw data of the time series

    """
    if isinstance(data, pd.Series):
        data_raw = data.values
        data_index = data.index
    elif isinstance(data, pd.DataFrame):
        data_raw = data.values
        data_index = data.columns
    else:
        data_raw = data
        data_index = np.arange(data.shape[-1])

    try:
        return (data_index.astype(np.float64), data_raw.astype(np.float64, copy=False))
    except TypeError:  # datetime index cannot be cast to float64
        return (data_index, data_raw.astype(np.float64, copy=False))


def read_dataset(dataset, sampling_factor=10000):
    """ Reads a dataset and resamples.

    Parameters
    ----------
    dataset : String
        File location.
    sampling_factor :
        The time series is sampled down to roughly this number of points by skipping
        every other point.

    Returns
    -------
    data : array-like
        The time series with z-score applied.

    """
    full_path = '../datasets/' + dataset
    data = pd.read_csv(full_path).T
    data = np.array(data)[0]
    print("Dataset Original Length n: ", len(data))

    data, factor = _resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    return zscore(data)


@njit(fastmath=True, cache=True)
def _sliding_dot_product(query, time_series):
    """Compute a sliding dot-product using the Fourier-Transform

    Parameters
    ----------
    query : array-like
        first time series, typically shorter than ts
    time_series : array-like
        second time series, typically longer than query.

    Returns
    -------
    dot_product : array-like
        The result of the sliding dot-product
    """

    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1

    query = query[::-1]

    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))

    trim = m - 1 + time_series_add

    with objmode(dot_product="float64[:]"):
        dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))

    return dot_product[trim:]


@njit(fastmath=True, cache=True, parallel=True)
def compute_distance_matrix(time_series,
                            m,
                            k,
                            exclude_trivial_match=True,
                            compute_knns=True,
                            n_jobs=4,
                            slack=0.5,
                            sum_dims=True,
                            distance=znormed_euclidean_distance,
                            distance_preprocessing=sliding_mean_std
                            ):
    """ Compute the full Distance Matrix between all pairs of subsequences of a
        multivariate time series.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted
        from the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        time_series : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool
            Trivial matches will be excluded if this parameter is set
        n_jobs : int
            Number of jobs to be used.
        slack: float
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        sum_dims : bool
            Sum distances overa ll dimensions into one row for
            multidimensional time series
        distance: callable
                The distance function to be computed.
        distance_preprocessing: callable
                The distance preprocessing function to be computed.

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """
    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    # Sum all dimensions into one row
    if sum_dims:
        D_all = np.zeros((1, n, n), dtype=np.float32)
        if compute_knns:
            knns = np.zeros((1, n, k), dtype=np.int32)
        else:
            knns = np.zeros((dims, 1, 1), dtype=np.int32)
    else:
        D_all = np.zeros((dims, n, n), dtype=np.float32)
        if compute_knns:
            knns = np.zeros((dims, n, k), dtype=np.int32)
        else:
            knns = np.zeros((dims, 1, 1), dtype=np.int32)

    bin_size = time_series.shape[-1] // n_jobs
    # lower_bounds = np.zeros(n_jobs)
    # lower_bounds[:] = np.inf

    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, time_series.shape[-1] - m + 1)

        for d in np.arange(dims):
            ts = time_series[d, :]
            # means, stds = _sliding_mean_std(ts, m)
            preprocessing = distance_preprocessing(ts, m)
            dot_first = _sliding_dot_product(ts[:m], ts)

            dot_prev = None
            for order in np.arange(start, end):
                if order == start:
                    # O(n log n) operation
                    dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    # constant time O(1) operations
                    dot_rolled = np.roll(dot_prev, 1) \
                                 + ts[order + m - 1] * ts[m - 1:n + m] \
                                 - ts[order - 1] * np.roll(ts[:n], 1)
                    dot_rolled[0] = dot_first[order]

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)
                if sum_dims:
                    D_all[0, order] += dist
                else:
                    D_all[d, order] = dist
                dot_prev = dot_rolled

        if sum_dims:
            D_all = D_all / dims

        if compute_knns:
            # do not merge with previous loop, as we are adding distances
            # over dimensions, first
            for d in np.arange(D_all.shape[0]):
                for order in np.arange(start, end):
                    knn = _argknn(D_all[d, order], k, m, slack=slack)

                    knns[d, order, :len(knn)] = knn
                    knns[d, order, len(knn):] = -1

                    # if len(knn) == k:
                    #    lower_bounds[idx] = min(lower_bounds[idx],
                    #                            4 * D_all[d, order, knn[-1]])

    return D_all, knns


@njit(fastmath=True, cache=True, parallel=True)
def compute_distance_matrix_sparse(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_preprocessing=sliding_mean_std
    ):
    """ Compute the full Distance Matrix between all pairs of subsequences of a
        multivariate time series.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted
        from the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        time_series : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool
            Trivial matches will be excluded if this parameter is set
        n_jobs : int
            Number of jobs to be used.
        slack: float
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        distance: callable
                The distance function to be computed.
        distance_preprocessing: callable
                The distance preprocessing function to be computed.

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """
    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    D_knn = np.zeros((dims, n, k), dtype=np.float32)
    knns = np.zeros((dims, n, k), dtype=np.int32)

    # TODO: no sparse matrix support in numba. Thus we use this hack
    # TODO set???
    D_bool = [Dict.empty(key_type=types.int32, value_type=types.bool_) for _ in
              range(n)]

    D_sparse = List()
    for d in range(dims):
        _list2 = List()
        D_sparse.append(_list2)
        for i in range(n):
            _list2.append(Dict.empty(key_type=types.int32, value_type=types.float32))

    lowest_distance = np.zeros(k, dtype=np.float32)
    lowest_distance[:] = np.inf

    # first pass, computing the k-nns
    for d in range(dims):
        ts = time_series[d, :]
        # means, stds = _sliding_mean_std(ts, m)
        preprocessing = distance_preprocessing(ts, m)

        dot_first = _sliding_dot_product(ts[:m], ts)
        bin_size = ts.shape[0] // n_jobs
        for idx in prange(n_jobs):
            start = idx * bin_size
            end = min((idx + 1) * bin_size, n)

            dot_prev = None
            for order in np.arange(start, end):
                if order == start:
                    # O(n log n) operation
                    dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    # constant time O(1) operations
                    dot_rolled = np.roll(dot_prev, 1) \
                                 + ts[order + m - 1] * ts[m - 1:n + m] \
                                 - ts[order - 1] * np.roll(ts[:n], 1)
                    dot_rolled[0] = dot_first[order]

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)
                dot_prev = dot_rolled

                knn = _argknn(dist, k, m, slack=slack)
                D_knn[d, order] = dist[knn]
                knns[d, order] = knn

    # compute a lower bound for the distance
    # for d in range(dims):
    #    for dist_knn in D_knn[d]:
    #        # np.minimum does not work with numba
    #        for i in range(len(dist_knn)):
    #            lowest_distance[i] = min(lowest_distance[i], dist_knn[i])

    # 2*r is the maximal distance between two subsequences
    # lowest_distance = 4 * lowest_distance * dims  # FIXME: depends on worst dimension

    # Parallelizm does not work, as Dict is not thread safe :(
    # FIXME : given a large number of dimensions, the memory usage explodes
    for d in np.arange(dims):
        for order in np.arange(0, n):
            for ks in knns[d, order]:  # needed to compute the k-nn distances
                D_bool[order][ks] = True

        for order in np.arange(0, n):
            # all pairs only needed when lower bound is given
            # if np.any(D_knn[d, order][1:] <= lowest_distance[1:]):
            # memorize which pairs are needed
            for ks in knns[d, order]:
                for ks2 in knns[d, order]:
                    D_bool[ks][ks2] = True

    # second pass, filling only the pairs needed
    for d in range(dims):
        ts = time_series[d, :]
        # means, stds = _sliding_mean_std(ts, m)
        preprocessing = distance_preprocessing(ts, m)

        dot_first = _sliding_dot_product(ts[:m], ts)
        bin_size = ts.shape[0] // n_jobs
        for idx in prange(n_jobs):
            start = idx * bin_size
            end = min((idx + 1) * bin_size, n)

            dot_prev = None
            for order in np.arange(start, end):
                if order == start:
                    # O(n log n) operation
                    dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    # constant time O(1) operations
                    dot_rolled = np.roll(dot_prev, 1) \
                                 + ts[order + m - 1] * ts[m - 1:n + m] \
                                 - ts[order - 1] * np.roll(ts[:n], 1)
                    dot_rolled[0] = dot_first[order]

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)
                dot_prev = dot_rolled

                # fill the knns now with the distances computed
                for key in D_bool[order]:
                    D_sparse[d][order][key] = dist[key]

    return D_knn, D_sparse, knns


@njit(fastmath=True, cache=True)
def get_radius(D_full, motifset_pos):
    """Computes the radius of the passed motif set (leitmotif).

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets

    Returns
    -------
    leitmotif_radius : float
        The radius of the motif set
    """
    leitmotif_radius = np.inf

    for ii in range(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        current = np.float32(0.0)
        for jj in range(1, len(motifset_pos)):
            if (i != jj):
                j = motifset_pos[jj]
                current = max(current, D_full[i, j])
        leitmotif_radius = min(current, leitmotif_radius)

    return leitmotif_radius


@njit(fastmath=True, cache=True)
def get_pairwise_extent(D_full, motifset_pos, dim_index, upperbound=np.inf):
    """Computes the extent of the motifset.

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets
    upperbound : float, default: np.inf
        Upper bound on the distances. If passed, will apply admissible pruning
        on distance computations, and only return the actual extent, if it is lower
        than `upperbound`

    Returns
    -------
    motifset_extent : float
        The extent of the motif set, if smaller than `upperbound`, else np.inf
    """

    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)
    # dimension chosen based on "first to k-th entry" order
    idx = dim_index[motifset_pos[-1]]

    for ii in range(len(motifset_pos) - 1):
        i = motifset_pos[ii]

        for jj in range(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]

            extent = np.float64(0.0)
            for kk in range(len(idx)):
                extent += D_full[idx[kk]][i][j]

            motifset_extent = max(motifset_extent, extent)
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(fastmath=True, cache=True)
def _argknn(
        dist, k, m, lowest_dist=np.inf, slack=0.5):
    """Finds the closest k-NN non-overlapping subsequences in candidates.

    Parameters
    ----------
    dist : array-like
        the distances
    k : int
        The k in k-NN
    m : int
        The window-length
    lowest_dist : float
        Used for admissible pruning
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    halve_m = int(m * slack)
    dists = np.copy(dist)

    dist_pos = np.argpartition(dist, 2 * k)[:2 * k]
    dist_sort = dist[dist_pos]

    idx = []  # there may be less than k, thus use a list

    # go through the partitioned list
    for i in range(len(dist_sort)):
        p = np.argmin(dist_sort)
        pos = dist_pos[p]
        dist_sort[p] = np.inf

        if (not np.isnan(dists[pos])) \
                and (not np.isinf(dists[pos])) \
                and (dists[pos] <= lowest_dist):
            idx.append(pos)

            # exclude all trivial matches and itself
            dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf

        if len(idx) == k:
            break

    # if not enough elements found, go through the rest
    for i in range(len(idx), k):
        pos = np.argmin(dists)
        if (not np.isnan(dists[pos])) \
                and (not np.isinf(dists[pos])) \
                and (dists[pos] <= lowest_dist):
            idx.append(pos)

            # exclude all trivial matches
            dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf
        else:
            break

    return np.array(idx, dtype=np.int32)


@njit(fastmath=True, cache=True)
def run_LAMA(
        ts, m, k, D, knns, dim_index, upper_bound=np.inf
):
    """Compute the approximate leitmotif using LAMA.

    Details are given within the paper Section 3.3
    LAtent leitMotif discovery Algorithm (LAMA).

    Parameters
    ----------
    ts : array-like
        The raw time seres
    m : int
        The motif length
    k : int
        The size k of the leitmotif
    D : 2d array-like
        The distance matrix
    upper_bound : float
        Used for admissible pruning

    Returns
    -------
    Tuple
        leitmotif_candidate : np.array
            The (approximate) best leitmotif found
        leitmotif_dist:
            The extent of the leitmotif found
    """
    n = ts.shape[-1] - m + 1
    leitmotif_dist = upper_bound
    leitmotif_dims = None
    leitmotif_candidate = None

    for order in np.arange(n, dtype=np.int32):
        # for dim in np.arange(ts.shape[0], dtype=np.int32):
        # knn_idx = knns[dim_index[order, dim], order]

        # Use the first (best) dimension for ordering of k-NNs
        knn_idx = knns[dim_index[order, 0], order]
        if np.any(knn_idx[:k] == -1):
            continue

        # sum over the knns from the best dimensions
        # TODO
        knn_distance = 0.0
        for a in np.arange(dim_index.shape[-1]):  # dimensions
            knn_distance += D[dim_index[order, a]][order][knn_idx[k - 1]]

        if len(knn_idx) >= k and knn_idx[k - 1] >= 0:
            if knn_distance <= leitmotif_dist:
                extent = get_pairwise_extent(
                    D,
                    knn_idx[:k],
                    dim_index,
                    leitmotif_dist
                )
                if extent <= leitmotif_dist:
                    leitmotif_dist = extent
                    leitmotif_candidate = knn_idx[:k]
                    leitmotif_dims = dim_index[order]

    # print("best dims", m, k, leitmotif_dims)
    return leitmotif_candidate, leitmotif_dist, leitmotif_dims


@njit(fastmath=True, cache=True)
def _check_unique(motifset_1, motifset_2, motif_length):
    """Check for overlaps between two motif sets.

    Two motif sets overlapp, if more than m/2 subsequences overlap from motifset 1.

    Parameters
    ----------
    motifset_1 : array-like
        Positions of the smaller motif set.
    motifset_2 : array-like
        Positions of the larger motif set.
    motif_length : int
        The length of the motif. Overlap exists, if 25% of two subsequences overlap.

    Returns
    -------
    True, if there are at least m/2 subsequences with an overlap of 25%, else False.
    """
    count = 0
    for a in motifset_1:  # smaller leitmotif
        for b in motifset_2:  # larger leitmotif
            if abs(a - b) < (motif_length / 4):
                count = count + 1
                break

        if count >= len(motifset_1) / 2:
            return False
    return True


# @njit(fastmath=True, cache=True)
def _filter_unique(elbow_points, candidates, motif_length):
    """Filters the list of candidate elbows for only the non-overlapping motifsets.

    This method applied a duplicate detection by filtering overlapping motif sets.
    Two candidate motif sets overlap, if at least m/2 subsequences of the smaller
    motifset overlapp with the larger motifset. Only the largest non-overlapping
    motif sets are retained.

    Parameters
    ----------
    elbow_points : array-like
        List of possible k's for elbow-points.
    candidates : 2d array-like
        List of motif sets for each k
    motif_length : int
        Length of the motifs, needed for checking overlaps.

    Returns
    -------
    filtered_ebp : array-like
        The set of non-overlapping elbow points.

    """
    filtered_ebp = []  # TODO
    for i in range(len(elbow_points)):
        unique = True
        for j in range(i + 1, len(elbow_points)):
            unique = _check_unique(
                candidates[elbow_points[i]], candidates[elbow_points[j]], motif_length)
            if not unique:
                break
        if unique:
            filtered_ebp.append(elbow_points[i])

    return np.array(filtered_ebp)


@njit(fastmath=True, cache=True)
def find_elbow_points(dists, alpha=2, elbow_deviation=1.00):
    """Finds elbow-points in the elbow-plot (extent over each k).

    Parameters
    ----------
    dists : array-like
        The extends for each k.
    alpha : float
        A threshold used to detect an elbow-point in the distances.
        It measures the relative change in deviation from k-1 to k to k+1.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    elbow_points : the elbow-points in the extent-function
    """
    elbow_points = set()
    elbow_points.add(2)  # required for numba to have a type
    elbow_points.clear()

    peaks = np.zeros(len(dists))
    for i in range(3, len(peaks) - 1):
        if (dists[i] != np.inf and
                dists[i + 1] != np.inf and
                dists[i - 1] != np.inf):

            m1 = (dists[i + 1] - dists[i]) + 0.00001
            m2 = (dists[i] - dists[i - 1]) + 0.00001

            # avoid detecting elbows in near constant data
            if dists[i - 1] == dists[i]:
                m2 = 1.0  # peaks[i] = 0

            if (dists[i] > 0) and (dists[i + 1] / dists[i] > elbow_deviation):
                peaks[i] = (m1 / m2)

    elbow_points = []
    while True:
        p = np.argmax(peaks)
        if peaks[p] > alpha:
            elbow_points.append(p)
            peaks[p - 1:p + 2] = 0
        else:
            break

    if len(elbow_points) == 0:
        elbow_points.append(2)

    return np.sort(np.array(list(set(elbow_points))))


def select_subdimensions(
        data,
        k_max,
        motif_length,
        dim_range,
        minimize_pairwise_dist=False,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_preprocessing=sliding_mean_std):
    """Findes the optimal number of dimensions

    Parameters
    ----------
    data : array-like
        The time series.
    k_max : int
        The maximum value of k's to compute the area of a single AU_EF.
    motif_length : int
        The length of the motif
    dim_range : list
        the range of dimensions to use for subdimensional motif discovery
    n_jobs : int
        Number of jobs to be used.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack : float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    distance: callable
        The distance function to be computed.
    distance_preprocessing: callable
        The distance preprocessing function to be computed.

    Returns
    -------
    Tuple
        minimum : array-like
            The minumum found
        all_minima : array-like
            All local minima found
        au_efs : array-like
            For each length in the interval, the AU_EF.
        elbows :
            Largest k (largest elbow) found
        top_leitmotifs :
            The leitmotif for the largest k for each length.

    """
    # in reverse order
    all_dist = np.zeros(len(dim_range), dtype=object)
    all_candidates = np.zeros(len(dim_range), dtype=object)
    all_candidate_dims = np.zeros(len(dim_range), dtype=object)
    all_elbow_points = np.zeros(len(dim_range), dtype=object)

    D_full = None
    knns = None
    for i, n_dims in enumerate(dim_range):
        if n_dims <= data.shape[0]:
            dist, candidates, candidate_dims, elbow_points, D_full, knns \
                = search_leitmotifs_elbow(
                k_max,
                data,
                motif_length,
                n_dims=n_dims,
                elbow_deviation=elbow_deviation,
                slack=slack,
                return_distances=True,
                minimize_pairwise_dist=minimize_pairwise_dist,
                D_full=D_full,
                knns=knns,  # reuse distances from last runs
                n_jobs=n_jobs,
                distance=distance,
                distance_preprocessing=distance_preprocessing
            )

            elbow_points = _filter_unique(elbow_points, candidates, motif_length)

            all_dist[i] = dist[elbow_points[-1]]
            all_candidates[i] = candidates[elbow_points[-1]]
            all_candidate_dims[i] = candidate_dims[elbow_points[-1]]
            all_elbow_points[i] = elbow_points[-1]

    return (all_dist,
            all_candidates,
            all_candidate_dims,
            all_elbow_points)


def find_au_ef_motif_length(
        data,
        k_max,
        motif_length_range,
        n_dims=None,
        minimize_pairwise_dist=False,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        subsample=2,
        distance=znormed_euclidean_distance,
        distance_preprocessing=sliding_mean_std
    ):
    """Computes the Area under the Elbow-Function within an of motif lengths.

    Parameters
    ----------
    data : array-like
        The time series.
    k_max : int
        The interval of k's to compute the area of a single AU_EF.
    motif_length_range : array-like
        The range of lengths to compute the AU-EF.
    n_dims : int
        the number of dimensions to use for subdimensional motif discovery
    n_jobs : int
        Number of jobs to be used.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    distance: callable
        The distance function to be computed.
    distance_preprocessing: callable
        The distance preprocessing function to be computed.

    Returns
    -------
    Tuple
        minimum : array-like
            The minumum found
        all_minima : array-like
            All local minima found
        au_efs : array-like
            For each length in the interval, the AU_EF.
        elbows :
            Largest k (largest elbow) found
        top_leitmotifs :
            The leitmotif for the largest k for each length.

    """
    # apply sampling for speedup only
    if subsample > 1:
        if data.ndim >= 2:
            data = data[:, ::subsample]
        else:
            data = data[::subsample]

    # in reverse order
    au_efs = np.zeros(len(motif_length_range), dtype=object)
    au_efs.fill(np.inf)
    elbows = np.zeros(len(motif_length_range), dtype=object)
    top_leitmotifs = np.zeros(len(motif_length_range), dtype=object)
    top_leitmotifs_dims = np.zeros(len(motif_length_range), dtype=object)
    dists = np.zeros(len(motif_length_range), dtype=object)

    # TODO parallelize?
    for i, m in enumerate(motif_length_range[::-1]):
        if m // subsample < data.shape[-1]:
            dist, candidates, candidate_dims, elbow_points, _ = search_leitmotifs_elbow(
                k_max,
                data,
                m // subsample,
                n_dims=n_dims,
                n_jobs=n_jobs,
                elbow_deviation=elbow_deviation,
                minimize_pairwise_dist=minimize_pairwise_dist,
                slack=slack,
                distance=distance,
                distance_preprocessing=distance_preprocessing
            )

            dists_ = dist[(~np.isinf(dist)) & (~np.isnan(dist))]
            # dists_ = dists_[:min(elbow_points[-1] + 1, len(dists_))]
            if dists_.max() - dists_.min() == 0:
                au_efs[i] = 1.0
            else:
                au_efs[i] = (((dists_ - dists_.min()) / (
                        dists_.max() - dists_.min())).sum()
                             / len(dists_))

            elbow_points = _filter_unique(elbow_points, candidates, m // subsample)

            if len(elbow_points > 0):
                elbows[i] = elbow_points
                top_leitmotifs[i] = candidates[elbow_points]
                top_leitmotifs_dims[i] = candidate_dims[elbow_points]
            else:
                # we found only the pair motif
                elbows[i] = [2]
                top_leitmotifs[i] = [candidates[2]]
                top_leitmotifs_dims[i] = candidate_dims[candidates[2]]

                # no elbow can be found, ignore this part
                au_efs[i] = 1.0

            dists[i] = dist

    # reverse order
    au_efs = np.array(au_efs, dtype=np.float64)[::-1]
    elbows = elbows[::-1]
    dists = dists[::-1]
    top_leitmotifs = top_leitmotifs[::-1] * subsample
    top_leitmotifs_dims = top_leitmotifs_dims[::-1]

    # Minima in AU_EF
    minimum = motif_length_range[np.nanargmin(au_efs)]
    au_ef_minima = argrelextrema(au_efs, np.less_equal, order=subsample)

    # Maxima in the EF
    return (minimum,
            au_ef_minima, au_efs,
            elbows,
            top_leitmotifs, top_leitmotifs_dims,
            dists)


def search_leitmotifs_elbow(
        k_max,
        data,
        motif_length,
        n_dims=None,
        elbow_deviation=1.00,
        filter=True,
        slack=0.5,
        return_distances=False,
        D_full=None,
        knns=None,
        minimize_pairwise_dist=False,
        n_jobs=4,
        distance=znormed_euclidean_distance,
        distance_preprocessing=sliding_mean_std
):
    """Computes the elbow-function.

    This is the method to find the characteristic leitmotifs within range
    [2...k_max] for given a `motif_length` using elbow-plots.

    Details are given within the paper Section 5.1 Learning meaningful k.

    Parameters
    ----------
    k_max : int
        use [2...k_max] to compute the elbow plot (user parameter).
    data : array-like
        the TS
    motif_length : int
        the length of the motif (user parameter)
    n_dims : int
        the number of dimensions to use for subdimensional motif discovery
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    filter: bool, default=True
        filters overlapping leitmotif from the result,
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    n_jobs : int
        Number of jobs to be used.
    distance: callable
            The distance function to be computed.
    distance_preprocessing: callable
            The distance preprocessing function to be computed.

    Returns
    -------
    Tuple
        dists :
            distances for each k in [2...k_max]
        candidates :
            motifset-candidates for each k
        elbow_points :
            elbow-points
        m : int
            best motif length
    """
    # convert to numpy array
    _, data_raw = pd_series_to_numpy(data)

    # m: motif size, n: number of subsequences, d: dimensions
    m = motif_length
    n = data_raw.shape[-1] - m + 1
    d = data_raw.shape[0]

    k_max_ = max(3, min(int(n // (m * slack)), k_max))

    # Check if use_dim is smaller than all given dimensions
    n_dims = d if n_dims is None else n_dims
    sum_dims = True if n_dims >= d else False

    # switch to sparse matrix representation when length is above 30_000
    # sparse matrix is 2x slower but needs less memory
    sparse_gb = ((n ** 2) * d) * 32 / (1024 ** 3) / 8
    sparse = sparse_gb > 8.0

    # compute the distance matrix
    if D_full is None:
        if minimize_pairwise_dist:  # FIXME: find better name
            # this has the drawback, that each pair of subsequences may
            # have different smallest dimensions

            print("Sort along dimension axis", flush=True)
            D_full, _ = compute_distance_matrix(
                data_raw, m, k_max_,
                compute_knns=False,
                n_jobs=n_jobs,
                slack=slack,
                sum_dims=False,
                distance=distance,
                distance_preprocessing=distance_preprocessing,
            )

            D_full = np.sort(D_full, axis=0)[:n_dims].sum(axis=0, dtype=np.float32)
            knns = _argknns(D_full, k_max_, m, n, slack)

            D_full = D_full.reshape(1, D_full.shape[0], D_full.shape[1])
            knns = knns.reshape(1, knns.shape[0], knns.shape[1])
        elif sparse:
            print("Using Sparse Backend", flush=True)
            D_knns, D_full, knns = compute_distance_matrix_sparse(
                data_raw, m, k_max_,
                n_jobs=n_jobs,
                slack=slack,
                distance=distance,
                distance_preprocessing=distance_preprocessing
            )
        else:
            # print("Using Default Backend", flush=True)
            D_full, knns = compute_distance_matrix(
                data_raw, m, k_max_,
                n_jobs=n_jobs,
                slack=slack,
                sum_dims=sum_dims,
                distance=distance,
                distance_preprocessing=distance_preprocessing
            )

    # non-overlapping motifs only
    k_leitmotif_distances = np.zeros(k_max_ + 1)
    k_leitmotif_candidates = np.empty(k_max_ + 1, dtype=object)
    k_leitmotif_dims = np.empty(k_max_ + 1, dtype=object)

    # order dimensions by increasing distance
    use_dim = min(n_dims, len(D_full))  # dimensions indexed by 0

    upper_bound = np.inf
    for test_k in tqdm(range(k_max_, 1, -1),
                       desc='Compute ks (' + str(k_max_) + ")",
                       position=0, leave=False):

        if minimize_pairwise_dist or sum_dims:
            # Do nothing
            dim_index = np.zeros((n, 1), dtype=np.int32)
        elif not sum_dims:
            # k-th NN and it's distance along all dimensions
            knn_idx = knns[:, :, test_k - 1]
            if isinstance(D_full, List) or isinstance(D_full, list):
                D_knn = take_along_axis(D_full, d, knn_idx, n)
            else:
                D_knn = np.take_along_axis(
                    D_full,
                    knn_idx.reshape((knn_idx.shape[0], knn_idx.shape[1], 1)),
                    axis=2)[:, :, 0]

            dim_index = np.argsort(D_knn, axis=0)[:use_dim]
            dim_index = np.transpose(dim_index, (1, 0))

        candidate, candidate_dist, candidate_dims = run_LAMA(
            data_raw, m, test_k, D_full, knns, dim_index,
            upper_bound=upper_bound,
        )

        k_leitmotif_distances[test_k] = candidate_dist
        k_leitmotif_candidates[test_k] = candidate

        if minimize_pairwise_dist or sum_dims:
            k_leitmotif_dims[test_k] = np.arange(d)
        elif not sum_dims:
            k_leitmotif_dims[test_k] = candidate_dims

        # compute a new upper bound
        # if candidate is not None:
        #    dist_new = get_pairwise_extent(D_full, candidate[:test_k])
        #    upper_bound = min(upper_bound, dist_new)
        # FIXME: upper_bound = min(candidate_dist, upper_bound)

    # smoothen the line to make it monotonically increasing
    k_leitmotif_distances[0:2] = k_leitmotif_distances[2]
    for i in range(len(k_leitmotif_distances), 2):
        k_leitmotif_distances[i - 1] = min(k_leitmotif_distances[i],
                                           k_leitmotif_distances[i - 1])

    elbow_points = find_elbow_points(k_leitmotif_distances,
                                     elbow_deviation=elbow_deviation)

    if filter:
        elbow_points = _filter_unique(
            elbow_points, k_leitmotif_candidates, motif_length)

    if return_distances:
        return (k_leitmotif_distances, k_leitmotif_candidates, k_leitmotif_dims,
                elbow_points, D_full, knns)
    else:
        return (k_leitmotif_distances, k_leitmotif_candidates, k_leitmotif_dims,
                elbow_points, m)


@njit(fastmath=True, cache=True)
def _argknns(D_full, k_max_, m, n, slack):
    # compute knns from new distance matrix
    knns = np.zeros((n, k_max_), dtype=np.int32)
    for order in range(0, D_full.shape[0]):
        knn = _argknn(D_full[order], k_max_, m, slack=slack)
        knns[order, :len(knn)] = knn
        knns[order, len(knn):] = -1

    return knns


@njit(fastmath=True, cache=True, parallel=True)
def take_along_axis(D_full, d, knn_idx, n):
    D_knn = np.zeros((d, n), dtype=np.float32)
    for i in prange(d):
        for j in prange(n):
            D_knn[i, j] = D_full[i][j][knn_idx[i, j]]
    return D_knn


@njit(fastmath=True, cache=True)
def candidate_dist(D_full, pool, upperbound, m, slack=0.5):
    leitmotif_candidate_dist = 0
    m_half = int(m * slack)
    for i in pool:
        for j in pool:
            if ((i != j and np.abs(i - j) < m_half)
                    or (i != j and D_full[i, j] > upperbound)):
                return np.inf

    for i in pool:
        for j in pool:
            leitmotif_candidate_dist = max(leitmotif_candidate_dist, D_full[i, j])

    return leitmotif_candidate_dist


# @njit(fastmath=True, cache=True, parallel=True)
def compute_distances_full_univ(ts, m, exclude_trivial_match=True, n_jobs=4, slack=0.5):
    """Compute the full Distance Matrix between all pairs of subsequences.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted
        from the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        ts : array-like
            The time series
        m : int
            The window length
        exclude_trivial_match : bool
            Trivial matches will be excluded if this parameter is set
        n_jobs : int
            Number of jobs to be used.
        slack: float
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences

    """
    return compute_distance_matrix(
        ts,
        m,
        1,
        exclude_trivial_match=exclude_trivial_match,
        n_jobs=n_jobs,
        slack=slack,
        sum_dims=True)[0]
