# -*- coding: utf-8 -*-
"""Discover subdimensional motif sets with LAMA.

This module contains the public :class:`LAMA` estimator plus the numerical
k-nearest-neighbor, extent, elbow, dimension-selection, and plotting-support
helpers used by the package. LAMA searches multidimensional time series for
repeated subsequences that are compact in a selected subset of dimensions.

The main workflow is:

* create ``LAMA(ds_name, series, n_dims=...)`` with dimensions on rows,
* optionally call ``fit_motif_length`` to choose a motif length,
* call ``fit_k_elbow`` to compute leitmotifs for ``k`` in ``[2, k_max]``.

By default, each ``k`` stores the single best leitmotif. Passing ``top_N > 1``
returns multiple non-overlapping leitmotifs per ``k``. The low-level search
keeps a rank axis for plotting, while ``LAMA.fit_k_elbow`` returns flattened,
row-aligned results in the same style as the Motiflets API.
"""

__author__ = ["patrickzib"]

import os
import warnings
from ast import literal_eval
from pathlib import Path

import pandas as pd
import psutil
from numba import set_num_threads, objmode, prange, get_num_threads
from numba import int32, float64
from numba.experimental import jitclass
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import argrelextrema
from scipy.stats import zscore

from leitmotifs.distances import *


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


_leitmotif_heap_spec = [
    ("heap_dist", float64[:]),
    ("heap_candidates", int32[:, :]),
    ("heap_dims", int32[:, :]),
    ("size", int32),
    ("capacity", int32),
]


@jitclass(_leitmotif_heap_spec)
class LeitmotifMaxHeap:
    """Fixed-size max-heap for leitmotif candidates and dimensions."""

    def __init__(self, capacity, k, n_dims):
        self.heap_dist = np.full(capacity, np.inf, dtype=np.float64)
        self.heap_candidates = np.full((capacity, k), -1, dtype=np.int32)
        self.heap_dims = np.full((capacity, n_dims), -1, dtype=np.int32)
        self.size = 0
        self.capacity = capacity

    def push(self, dist, candidate, dims):
        i = self.size
        self.heap_dist[i] = dist
        self.heap_candidates[i] = candidate
        self.heap_dims[i] = dims
        self.size += 1
        self._sift_up(i)

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap_dist[parent] >= self.heap_dist[i]:
                break
            self._swap(parent, i)
            i = parent

    def _sift_down(self, i):
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i

            if left < self.size and self.heap_dist[left] > self.heap_dist[largest]:
                largest = left
            if right < self.size and self.heap_dist[right] > self.heap_dist[largest]:
                largest = right

            if largest == i:
                break

            self._swap(i, largest)
            i = largest

    def _swap(self, a, b):
        self.heap_dist[a], self.heap_dist[b] = self.heap_dist[b], self.heap_dist[a]

        tmp_candidate = self.heap_candidates[a].copy()
        self.heap_candidates[a] = self.heap_candidates[b]
        self.heap_candidates[b] = tmp_candidate

        tmp_dims = self.heap_dims[a].copy()
        self.heap_dims[a] = self.heap_dims[b]
        self.heap_dims[b] = tmp_dims

    def sorted_entries(self):
        sorted_dists = np.full(self.capacity, np.inf, dtype=np.float64)
        sorted_candidates = np.full(self.heap_candidates.shape, -1, dtype=np.int32)
        sorted_dims = np.full(self.heap_dims.shape, -1, dtype=np.int32)
        if self.size == 0:
            return sorted_candidates, sorted_dists, sorted_dims

        order = np.argsort(self.heap_dist[:self.size])
        for i in range(self.size):
            sorted_candidates[i] = self.heap_candidates[order[i]]
            sorted_dists[i] = self.heap_dist[order[i]]
            sorted_dims[i] = self.heap_dims[order[i]]
        return sorted_candidates, sorted_dists, sorted_dims

    def replace_at(self, position, dist, candidate, dims):
        self.heap_dist[position] = dist
        self.heap_candidates[position] = candidate
        self.heap_dims[position] = dims
        self._sift_down(position)


def _strip_dataset_prefix(path):
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == ".." and parts[1] == "datasets":
        return Path(*parts[2:])
    if len(parts) >= 1 and parts[0] == "datasets":
        return Path(*parts[1:])
    return Path(path)


def _resolve_dataset_path(*parts):
    path = Path(*parts)
    if path.is_absolute() or path.exists():
        return path

    dataset_path = _strip_dataset_prefix(path)
    legacy_path = Path("..", "datasets", dataset_path)
    if legacy_path.exists():
        return legacy_path
    return _PROJECT_ROOT / "datasets" / dataset_path


def _resample(
        data,
        sampling_factor=10000
):
    """Resamples a time series to roughly `sampling_factor` points.

    The method searches a factor to skip every i-th point.

    Parameters
    ----------
    data : array-like
        The time series data
    sampling_factor : int (default=10000)
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


def read_ground_truth(dataset, path=None):
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
    dataset = Path(path or "") / Path(dataset)
    if "_gt.csv" not in str(dataset):
        file = Path(os.path.splitext(str(dataset))[0] + "_gt.csv")
    else:
        file = dataset

    file = _resolve_dataset_path(file)
    if file.exists():
        series = pd.read_csv(file, index_col=0)

        for i in range(0, series.shape[0]):
            series.iloc[i] = series.iloc[i].apply(literal_eval)

        return series
    else:
        print("No ground truth found for ", dataset)
    return None


def read_audio_from_dataframe(pandas_file_url, channels=None):
    """Reads a time series with an index (e.g. time) from a CSV with MFCC features."""
    pandas_file_url = _resolve_dataset_path(pandas_file_url)
    df = pd.read_csv(pandas_file_url, index_col=0, compression='gzip')
    audio_length_seconds = 2 * float(df.columns[-1]) - float(df.columns[-2])

    if channels:
        df = df.loc[channels]

    df_gt = read_ground_truth(pandas_file_url)

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
    full_path = _resolve_dataset_path("ground_truth", dataset)
    data = pd.read_csv(full_path, index_col=0).squeeze('columns')
    print("Dataset Original Length n: ", len(data))

    data, factor = _resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    data[:] = zscore(data)

    gt = read_ground_truth(full_path)
    if gt is not None:
        if factor > 1:
            for column in gt:
                gt[column] = gt[column].transform(lambda l: (np.array(l)) // factor)
        return data, gt
    else:
        return data


def convert_to_2d(
        series
):
    if series.ndim == 1:
        print('Warning: The input dimension must be 2d.')
        if isinstance(series, pd.Series):
            series = series.to_frame().T
        elif isinstance(series, (np.ndarray, np.generic)):
            series = series.reshape(1, -1)
    if series.shape[0] > series.shape[1]:
        raise ValueError(
            'Warning: The input shape is wrong. Dimensions should be on rows. '
            'Try transposing the input.')

    return series


def as_series(
        data,
        index_range,
        index_name):
    """Coverts a time series to a series with an index.

    Parameters
    ----------
    data : array-like
        The time series raw data as numpy array
    index_range :
        The index to use
    index_name :
        The name of the index to use (e.g. time)

    Returns
    -------
    series : PD.Series

    """
    series = pd.Series(data=data, index=index_range)
    series.index.name = index_name
    return series


def _filter_duplicate_window_sizes(au_ef, minima):
    """Filter neighboring window sizes with equal minima."""
    filtered = []
    pos = minima[0][0]
    last = au_ef[pos]
    for m in range(1, len(minima[0])):
        current = au_ef[minima[0][m]]
        if current != last:
            filtered.append(pos)
        last = current
        pos = minima[0][m]
    filtered.append(pos)
    return [np.array(filtered)]


def _plotting():
    import leitmotifs.plotting as plotting
    return plotting


class LAMA:
    """User-facing API for LAMA leitmotif discovery.

    Parameters
    ----------
    ds_name : str
        Name of the dataset, used in plot titles.
    series : array-like
        Time series data with dimensions on rows and time on columns.
    minimize_pairwise_dist : bool, default=False
        If True, each pairwise distance is minimized over dimensions before
        leitmotif search. This is similar to the mStamp approach and can select
        different dimensions for different subsequence pairs.
    ground_truth : pd.Series, optional
        Ground-truth intervals used only for plotting/evaluation helpers.
    dimension_labels : array-like, optional
        Labels for dimensions in plots. Numeric indices are used when omitted.
    elbow_deviation : float, default=1.00
        Minimum relative increase in the extent function for elbow detection.
    n_dims : int, optional
        Number of dimensions to select for subdimensional discovery. If omitted
        or greater than the data dimensionality, all dimensions are used.
    distance : str, default="znormed_ed"
        Distance family. Common values are ``"znormed_ed"`` and ``"ed"``.
    n_jobs : int, default=-1
        Number of Numba threads for distance computation. Values below 1 use
        all available CPU cores.
    slack : float, default=0.5
        Exclusion-zone width as a fraction of the motif length.
    backend : {"default", "scalable"}, default="default"
        Distance backend. ``"default"`` materializes full distance matrices;
        ``"scalable"`` keeps only nearest-neighbor distances and recomputes
        extents from raw data.
    """

    def __init__(
            self,
            ds_name,
            series,
            minimize_pairwise_dist=False,
            ground_truth=None,
            dimension_labels=None,
            elbow_deviation=1.00,
            n_dims=None,
            distance="znormed_ed",
            n_jobs=-1,
            slack=0.5,
            backend="default"
    ) -> None:
        self.ds_name = ds_name
        self.series = convert_to_2d(series)

        self.elbow_deviation = elbow_deviation
        self.slack = slack
        self.dimension_labels = dimension_labels
        self.ground_truth = ground_truth
        self.minimize_pairwise_dist = minimize_pairwise_dist

        # distance function used
        self.distance_preprocessing, self.distance, self.distance_single \
            = map_distances(distance)
        self.backend = backend

        self.motif_length_range = None
        self.motif_length = 0
        self.all_extrema = []
        self.all_elbows = []
        self.all_top_leitmotifs = []
        self.all_dists = []

        self.n_dims = n_dims

        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.n_jobs = n_jobs

        self.motif_length = 0
        self.memory_usage = 0
        self.k_max = 0
        self.dists = []
        self.leitmotifs = []
        self.elbow_points = []
        self.leitmotifs_dims = []
        self.all_dimensions = []
        self.top_n_flattened = False

    def fit_motif_length(
            self,
            k_max,
            motif_length_range,
            subsample=1,
            plot=True,
            plot_elbows=False,
            plot_motifsets=True,
            plot_best_only=True
    ):
        self.motif_length_range = motif_length_range
        self.k_max = k_max

        data = convert_to_2d(self.series)
        index, data_raw = pd_series_to_numpy(data)
        header = " in " + data.index.name if isinstance(
            data, pd.Series) and data.index.name is not None else ""
        motif_length_range = np.int32(motif_length_range)

        (self.motif_length,
         all_minima, au_ef,
         self.all_elbows,
         self.all_top_leitmotifs,
         self.all_dimensions,
         self.all_dists) = find_au_ef_motif_length(
            data_raw, k_max,
            n_dims=self.n_dims,
            motif_length_range=motif_length_range,
            minimize_pairwise_dist=self.minimize_pairwise_dist,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            subsample=subsample,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend
        )

        all_minima = _filter_duplicate_window_sizes(au_ef, all_minima)

        if plot:
            plotting = _plotting()
            plotting._plot_window_lengths(
                all_minima, au_ef, data_raw, self.ds_name, self.all_elbows,
                header, index, motif_length_range, self.all_top_leitmotifs,
                top_leitmotifs_dims=self.all_dimensions)

            if plot_elbows or plot_motifsets:
                to_plot = all_minima[0]
                if plot_best_only:
                    to_plot = [np.argmin(au_ef)]

                for a in to_plot:
                    motif_length = motif_length_range[a]
                    candidates = np.zeros(len(self.all_dists[a]), dtype=object)
                    candidates[self.all_elbows[a]] = self.all_top_leitmotifs[a]

                    candidate_dims = np.zeros(len(self.all_dists[a]), dtype=object)
                    candidate_dims[self.all_elbows[a]] = self.all_dimensions[a]

                    elbow_points = self.all_elbows[a]

                    if plot_elbows:
                        plotting._plot_elbow_points(
                            self.ds_name, data,
                            elbow_points, candidates, self.all_dists[a])

                    if plot_motifsets:
                        plotting.plot_motifsets(
                            self.ds_name,
                            data,
                            motifsets=self.all_top_leitmotifs[a],
                            leitmotif_dims=self.all_dimensions[a],
                            motif_length=motif_length,
                            ground_truth=self.ground_truth,
                            show=True)

        best_pos = np.argmin(au_ef)
        self.elbow_points = self.all_elbows[best_pos]
        self.dists = self.all_dists[best_pos]
        self.leitmotifs = np.zeros(len(self.all_dists[best_pos]), dtype=object)
        self.leitmotifs[self.all_elbows[best_pos]] = self.all_top_leitmotifs[best_pos]
        self.leitmotifs_dims = np.zeros(len(self.all_dists[best_pos]), dtype=object)
        self.leitmotifs_dims[self.all_elbows[best_pos]] = self.all_dimensions[best_pos]
        self.all_extrema = all_minima[0]

        return self.motif_length, self.all_extrema

    def fit_k_elbow(
            self,
            k_max,
            motif_length=None,  # if None, use best_motif_length
            filter_duplicates=True,
            plot_elbows=True,
            plot_motifsets=True,
            top_N=None,
    ):
        """Compute leitmotifs across candidate set sizes and detect elbows.

        Parameters
        ----------
        k_max : int
            Largest motif-set size to test. The effective value can be lower
            when the time series and exclusion zone cannot support that many
            non-overlapping occurrences.
        motif_length : int, optional
            Subsequence length. If omitted, the value learned by
            ``fit_motif_length`` is used.
        filter_duplicates : bool, default=True
            Remove elbow points whose leitmotifs overlap larger elbow motifs.
        plot_elbows : bool, default=True
            Plot the extent curve and detected elbow points.
        plot_motifsets : bool, default=True
            Plot discovered motif sets.
        top_N : int, optional
            Number of ranked, non-overlapping leitmotifs to retain per tested
            ``k``. ``None`` is equivalent to ``1`` and preserves the historical
            one-result return shape.

        Returns
        -------
        tuple
            ``(dists, leitmotifs, elbow_points)``. For ``top_N`` omitted or
            ``1``, ``dists[k]`` and ``leitmotifs[k]`` describe the best
            leitmotif for each ``k``. For ``top_N > 1``, the returned arrays
            are flattened and aligned by row; duplicate values in
            ``elbow_points`` indicate multiple ranks for the same ``k``.
        """
        self.k_max = k_max

        if motif_length is None:
            motif_length = self.motif_length
            if motif_length <= 0:
                raise ValueError(
                    "motif_length must be provided, or fit_motif_length() "
                    "must be called first.")
        else:
            self.motif_length = motif_length

        data = convert_to_2d(self.series)
        _, raw_data = pd_series_to_numpy(data)

        (dists, leitmotifs, leitmotifs_dims,
         elbow_points, _, self.memory_usage) = search_leitmotifs_elbow(
            k_max,
            raw_data,
            motif_length,
            n_dims=self.n_dims,
            filter=filter_duplicates,
            minimize_pairwise_dist=self.minimize_pairwise_dist,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend,
            top_N=top_N,
        )
        self.dists = dists
        self.leitmotifs = leitmotifs
        self.leitmotifs_dims = leitmotifs_dims
        self.elbow_points = elbow_points

        return_flattened = top_N is not None and top_N > 1
        self.top_n_flattened = False
        if return_flattened:
            flat_dists, flat_leitmotifs, flat_dims, flat_elbow_points = (
                flatten_elbows(
                    elbow_points, leitmotifs, leitmotifs_dims, dists,
                    max_items=None)
            )

        if plot_elbows or plot_motifsets:
            plotting = _plotting()
            if return_flattened:
                plotting.plot_elbow_result(
                    data=data,
                    ds_name=self.ds_name,
                    motif_length=motif_length,
                    dists=self.dists,
                    candidates=self.leitmotifs,
                    candidate_dims=self.leitmotifs_dims,
                    elbow_points=self.elbow_points,
                    show_elbows=plot_elbows,
                    show_grid=plot_motifsets,
                    ground_truth=self.ground_truth,
                    top_N=top_N)
                self.dists = flat_dists
                self.leitmotifs = flat_leitmotifs
                self.leitmotifs_dims = flat_dims
                self.elbow_points = flat_elbow_points
                self.top_n_flattened = True
                return self.dists, self.leitmotifs, self.elbow_points

            if plot_elbows:
                plotting._plot_elbow_points(
                    self.ds_name, data, self.elbow_points,
                    self.leitmotifs, self.dists)

            if plot_motifsets:
                motifsets = self.leitmotifs[self.elbow_points]
                motif_dims = self.leitmotifs_dims[self.elbow_points]
                if top_N is not None and top_N > 1:
                    motifsets = np.array(
                        [ranked[0] for ranked in motifsets], dtype=object)
                    motif_dims = np.array(
                        [ranked[0] for ranked in motif_dims], dtype=object)
                plotting.plot_motifsets(
                    self.ds_name,
                    data,
                    motifsets=motifsets,
                    leitmotif_dims=motif_dims,
                    motif_length=motif_length,
                    ground_truth=self.ground_truth,
                    show=True)

        if return_flattened:
            self.dists = flat_dists
            self.leitmotifs = flat_leitmotifs
            self.leitmotifs_dims = flat_dims
            self.elbow_points = flat_elbow_points
            self.top_n_flattened = True

        return self.dists, self.leitmotifs, self.elbow_points

    def fit_dimensions(
            self,
            k_max,
            motif_length,
            dim_range
    ):

        all_dist, all_candidates, all_candidate_dims, all_elbow_points \
            = select_subdimensions(
            self.series,
            k_max=k_max,
            motif_length=motif_length,
            dim_range=dim_range,
            minimize_pairwise_dist=self.minimize_pairwise_dist,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend
        )

        plotting = _plotting()

        fig, ax = plotting.plt.subplots(figsize=(10, 4))
        ax.set_title("Dimension Plot")
        plotting.sns.lineplot(x=dim_range, y=all_dist, ax=ax)
        plotting.plt.tight_layout()
        plotting.plt.show()

    def plot_dataset(self, path=None):
        fig, ax = _plotting().plot_dataset(
            self.ds_name,
            self.series,
            show=path is None,
            ground_truth=self.ground_truth)

        if path is not None:
            _plotting().plt.savefig(path)
            _plotting().plt.show()

        return fig, ax

    def plot_motifset(
            self,
            elbow_points=None,
            path=None,
            motifset_name=None):

        if self.dists is None or self.leitmotifs is None or self.elbow_points is None:
            raise Exception("Please call fit_k_elbow first.")

        if elbow_points is None:
            elbow_points = self.elbow_points

        # TODO
        # if elbow_point is None:
        #    elbow_point = self.elbow_points[-1]
        motifset_names = None
        if motifset_name is not None:
            motifset_names = [motifset_name for _ in range(len(self.elbow_points))]

        if self.top_n_flattened and len(self.leitmotifs) == len(elbow_points):
            motifsets = self.leitmotifs
            motif_dims = self.leitmotifs_dims
        else:
            motifsets = self.leitmotifs[elbow_points]
            motif_dims = self.leitmotifs_dims[elbow_points]
        if np.ndim(self.dists) == 2:
            motifsets = np.array([ranked[0] for ranked in motifsets], dtype=object)
            motif_dims = np.array([ranked[0] for ranked in motif_dims], dtype=object)

        fig, ax = _plotting().plot_motifsets(
            self.ds_name,
            self.series,
            motifsets=motifsets,
            leitmotif_dims=motif_dims,
            motifset_names=motifset_names,
            # dist=self.dists[elbow_points],
            ground_truth=self.ground_truth,
            motif_length=self.motif_length,
            show=path is None)

        if path is not None:
            _plotting().plt.savefig(path)
            _plotting().plt.show()

        return fig, ax


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
    full_path = _resolve_dataset_path(dataset)
    data = pd.read_csv(full_path).T
    data = np.array(data)[0]
    print("Dataset Original Length n: ", len(data))

    data, factor = _resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    return zscore(data)


@njit(cache=True)
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
    if m > n:
        raise ValueError("query longer than time_series")

    # Reverse query for cross-correlation.
    q_rev = query[::-1]

    with objmode(conv='float64[:]'):
        fft_length = next_fast_len(n + m - 1, real=True)
        conv = irfft(rfft(q_rev, fft_length) * rfft(time_series, fft_length),
                     fft_length)

    # Trim to the valid sliding-dot range
    return conv[m - 1: n]


@njit(fastmath=True, cache=True, inline='always')
def _update_sliding_dot_product(dot_rolled, dot_first_order, ts, order, m, n):
    add = ts[order + m - 1]
    remove = ts[order - 1]

    dot_rolled[1:] = (
            dot_rolled[:-1]
            + add * ts[m:n + m - 1]
            - remove * ts[:n - 1]
    )
    dot_rolled[0] = dot_first_order


@njit(parallel=True)
def compute_distances_with_knns_full(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        compute_knns=True,
        n_jobs=4,
        slack=0.5,
        sum_dims=True,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
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
    n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts

    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    # Sum all dimensions into one row
    if sum_dims:
        D_all = np.zeros((1, n, n), dtype=np.float32)
        if compute_knns:
            knns = np.full((1, n, k), -1, dtype=np.int32)
        else:
            knns = np.full((dims, 1, 1), -1, dtype=np.int32)
    else:
        D_all = np.zeros((dims, n, n), dtype=np.float32)
        if compute_knns:
            knns = np.full((dims, n, k), -1, dtype=np.int32)
        else:
            knns = np.full((dims, 1, 1), -1, dtype=np.int32)

    bin_size = np.int32(np.ceil(n / n_jobs))

    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min(start + bin_size, n)

        for d in np.arange(dims):
            ts = time_series[d, :]
            preprocessing = distance_preprocessing(ts, m)
            dot_first = _sliding_dot_product(ts[:m], ts)
            dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
            for order in np.arange(start, end):
                if order != start:
                    _update_sliding_dot_product(
                        dot_rolled, dot_first[order], ts, order, m, n)

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)

                if sum_dims:
                    D_all[0, order] += dist
                else:
                    D_all[d, order] = dist

    if compute_knns:
        # do not merge with previous loop, as we are adding distances
        # over dimensions, first
        for idx in prange(n_jobs):
            start = idx * bin_size
            end = min(start + bin_size, n)
            for d in np.arange(D_all.shape[0]):
                for order in np.arange(start, end):
                    knn = _argknn(D_all[d, order], k, m, slack=slack)
                    knns[d, order, :len(knn)] = knn

    if sum_dims:
        D_all = D_all / dims

    return D_all, knns


@njit(nogil=True, parallel=True)
def compute_distances_with_knns(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std
):
    """ Compute the full Distance Matrix between all pairs of subsequences of a
        multivariate time series.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted
        from the time series, of length n.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        time_series : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool (default: True)
            Trivial matches will be excluded if this parameter is set
        n_jobs : int (default: 4)
            Number of jobs to be used.
        slack: float (default: 0.5)
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        distance: callable (default: znormed_euclidean_distance)
                The distance function to be computed.
        distance_preprocessing: callable (default: sliding_mean_std)
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
    n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts

    halve_m = 0
    if exclude_trivial_match:
        halve_m = np.int32(m * slack)

    D_knn = np.zeros((dims, n, k), dtype=np.float64)
    knns = np.full((dims, n, k), -1, dtype=np.int32)

    bin_size = np.int32(np.ceil(n / n_jobs))

    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min(start + bin_size, n)

        for d in np.arange(dims):
            ts = time_series[d, :]
            preprocessing = distance_preprocessing(ts, m)
            dot_first = _sliding_dot_product(ts[:m], ts)
            dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
            for order in np.arange(start, end):
                if order != start:
                    _update_sliding_dot_product(
                        dot_rolled, dot_first[order], ts, order, m, n)

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)

                knn = _argknn(dist, k, m, slack=slack)
                knns[d, order, :len(knn)] = knn
                D_knn[d, order] = dist[knn]

    return D_knn, knns


@njit(cache=True)
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
        for jj in range(0, len(motifset_pos)):
            if (i != jj):
                j = motifset_pos[jj]
                current = max(current, D_full[i, j])
        leitmotif_radius = min(current, leitmotif_radius)

    return leitmotif_radius


@njit(cache=True)
def get_pairwise_extent(D_full, motifset_pos, dims, upperbound=np.inf):
    """Computes the extent of the motifset.

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets
    dims : array-like
        The sub-dimension to use
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

    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]

        for jj in range(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]

            extent = np.float64(0.0)
            for kk in range(len(dims)):
                extent += D_full[dims[kk]][i][j]

            motifset_extent = max(motifset_extent, extent)
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(cache=True, nogil=True)
def get_pairwise_extent_raw(
        series, motifset_pos, dims, motif_length,
        distance_single, preprocessing, upperbound=np.inf):
    """Computes the extent of the motifset via pairwise comparisons.

    Parameters
    ----------
    series : array-like
        The time series
    motifset_pos : array-like
        The motif set start-offsets
    dims : array-like
        The sub-dimension to use
    motif_length : int
        The motif length
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

    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        a = series[:, i:i + motif_length]

        for jj in np.arange(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]
            b = series[:, j:j + motif_length]

            extent = np.float64(0.0)
            for dim in dims:
                extent += distance_single(a[dim], b[dim], i, j, preprocessing[dim])

            motifset_extent = max(motifset_extent, extent)
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit
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
    lowest_dist : float (default=np.inf)
        Used for admissible pruning
    slack: float (default=0.5)
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    halve_m = np.int32(m * slack)
    dists = np.copy(dist)

    new_k = np.int32(min(len(dist) - 1, 2 * k))
    dist_pos = np.argpartition(dist, new_k)[:new_k]
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


@njit(cache=True)
def run_LAMA_top_n(
        ts, m, k, D, knns, dim_index,
        distance_single=None,
        preprocessing=None,
        use_D_full=True,
        upper_bound=np.inf,
        top_N=1
):
    """Return ranked approximate leitmotifs for a fixed motif size ``k``.

    The first rank is always the same best candidate returned by ``run_LAMA``.
    Additional ranks are selected from non-overlapping candidates using a
    fixed-size max-heap. Distinctness is based on temporal overlap of candidate
    positions; dimensions are stored as payload for each retained rank.
    """
    n = ts.shape[-1] - m + 1
    heap = LeitmotifMaxHeap(top_N, k, dim_index.shape[1])
    best_dist = upper_bound
    best_candidate = np.full(k, -1, dtype=np.int32)
    best_dims = np.full(dim_index.shape[1], -1, dtype=np.int32)

    for order in range(n):
        # Use the first (best) dimension for ordering of k-NNs
        order_dims = dim_index[order]
        knn_idx = knns[order_dims[0], order]
        if np.any(knn_idx[:k] == -1):
            continue

        kth = knn_idx[k - 1]

        # sum over the knns from the best dimensions
        knn_distance = np.float64(0.0)
        bound_check = upper_bound
        if heap.size == top_N:
            bound_check = heap.heap_dist[0]

        for d in order_dims:
            if use_D_full:
                knn_distance += D[d][order][kth]
            else:
                a = ts[d, order:order + m]
                b = ts[d, kth:kth + m]
                knn_distance += distance_single(a, b, order, kth, preprocessing[d])

            if knn_distance > bound_check:
                break

        if knn_distance <= bound_check:
            # dimension chosen based on "first to k-th entry" order
            candidate = knn_idx[:k]
            candidate_dims = dim_index[candidate[-1]]
            if use_D_full:
                candidate_extent = get_pairwise_extent(
                    D, candidate, candidate_dims, bound_check)
            else:
                candidate_extent = get_pairwise_extent_raw(
                    ts, candidate, candidate_dims,
                    m, distance_single, preprocessing, bound_check)

            if candidate_extent <= bound_check:
                if candidate_extent <= best_dist:
                    best_dist = candidate_extent
                    best_candidate = candidate.copy()
                    best_dims = candidate_dims.copy()

                overlap_count = 0
                overlap_pos = -1
                for j in range(heap.size):
                    if not _check_unique(candidate, heap.heap_candidates[j], m):
                        overlap_count += 1
                        overlap_pos = j

                if overlap_count == 0 and heap.size < top_N:
                    heap.push(candidate_extent, candidate, candidate_dims)
                elif overlap_count <= 1:
                    replace_pos = overlap_pos
                    if replace_pos == -1:
                        replace_pos = 0
                    if candidate_extent < heap.heap_dist[replace_pos]:
                        heap.replace_at(
                            replace_pos, candidate_extent, candidate, candidate_dims)

    sorted_candidates, sorted_dists, sorted_dims = heap.sorted_entries()
    if best_candidate[0] < 0:
        return sorted_candidates, sorted_dists, sorted_dims

    output_candidates = np.full(sorted_candidates.shape, -1, dtype=np.int32)
    output_dists = np.full(sorted_dists.shape, np.inf, dtype=np.float64)
    output_dims = np.full(sorted_dims.shape, -1, dtype=np.int32)

    output_candidates[0] = best_candidate
    output_dists[0] = best_dist
    output_dims[0] = best_dims
    output_size = 1

    for i in range(top_N):
        if output_size == top_N:
            break
        if sorted_candidates[i, 0] < 0:
            continue

        unique = True
        for j in range(output_size):
            if not _check_unique(sorted_candidates[i], output_candidates[j], m):
                unique = False
                break

        if unique:
            output_candidates[output_size] = sorted_candidates[i]
            output_dists[output_size] = sorted_dists[i]
            output_dims[output_size] = sorted_dims[i]
            output_size += 1

    return output_candidates, output_dists, output_dims


@njit(cache=True)
def run_LAMA(
        ts, m, k, D, knns, dim_index,
        distance_single=None,
        preprocessing=None,
        use_D_full=True,
        upper_bound=np.inf
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
    knns : 2d array-like
        The k-nns for each subsequence
    dim_index : 2d array-like
        The best dimensions
    use_D_full : bool
        If True, uses the full distance matrix D for computing the extent of the motiflet.
        If False, uses pairwise distances computed from the time series.
    upper_bound : float (default=np.inf)
        Used for admissible pruning

    Returns
    -------
    Tuple
        leitmotif_candidate : np.array
            The (approximate) best leitmotif found
        leitmotif_dist:
            The candidate_extent of the leitmotif found
    """
    candidates, dists, dims = run_LAMA_top_n(
        ts, m, k, D, knns, dim_index,
        distance_single=distance_single,
        preprocessing=preprocessing,
        use_D_full=use_D_full,
        upper_bound=upper_bound,
        top_N=1)

    return candidates[0], dists[0], dims[0]


@njit(cache=True)
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
    filtered_ebp = []
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


def flatten_elbows(elbow_points, candidates, candidate_dims, dists, max_items=None):
    """Flatten elbow/rank-aware leitmotif results for plotting or tabular use."""
    if dists is None or np.ndim(dists) != 2:
        return (
            np.array(dists, dtype=np.float64),
            np.array(candidates, dtype=object),
            np.array(candidate_dims, dtype=object),
            np.array(elbow_points, dtype=np.int32),
        )

    items = []
    for k in elbow_points:
        if candidates[k] is None:
            continue
        for rank in range(dists.shape[1]):
            if np.isinf(dists[k, rank]) or np.isnan(dists[k, rank]):
                continue
            if candidates[k][rank, 0] < 0:
                continue
            items.append((k, rank, dists[k, rank]))

    if max_items is not None:
        items.sort(key=lambda item: (-item[0], item[1], item[2]))
        items = items[:max_items]

    flat_candidates = []
    flat_dims = []
    flat_dists = []
    flat_elbows = []
    for k, rank, dist in items:
        flat_candidates.append(candidates[k][rank])
        flat_dims.append(candidate_dims[k][rank])
        flat_dists.append(dist)
        flat_elbows.append(k)

    return (
        np.array(flat_dists, dtype=np.float64),
        np.array(flat_candidates, dtype=object),
        np.array(flat_dims, dtype=object),
        np.array(flat_elbows, dtype=np.int32),
    )


@njit(cache=True)
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
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend='default'):
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
    minimize_pairwise_dist: bool (default=False)
        If True, the pairwise distance is minimized. This is the mStamp-approach.
        It has the potential drawback, that each pair of subsequences may have
        different smallest dimensions.
    n_jobs : int (default=4)
        Number of jobs to be used.
    elbow_deviation : float (default=1.00)
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack : float (default=0.5)
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    distance: callable (default=znormed_euclidean_distance)
        The distance function to be computed.
    distance_preprocessing: callable (default=sliding_mean_std)
        The distance preprocessing function to be computed.
    backend : String, default="scalable"
        The backend to use. As of now 'scalable' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory.

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
            dist, candidates, candidate_dims, elbow_points, D_full, knns, _ \
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
                distance_single=distance_single,
                distance_preprocessing=distance_preprocessing,
                backend=backend
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
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend='default'
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
    backend : String, default="scalable"
        The backend to use. As of now 'scalable' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory.

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
    motif_length_range = np.asarray(motif_length_range, dtype=np.int32)
    if motif_length_range.size == 0:
        raise ValueError("motif_length_range must contain at least one length.")

    invalid_lengths = motif_length_range[
        (motif_length_range <= 0) | (motif_length_range >= data.shape[-1])
    ]
    if invalid_lengths.size > 0:
        raise ValueError(
            "motif_length_range values must be positive and smaller than "
            f"the time series length ({data.shape[-1]}). Invalid lengths: "
            f"{invalid_lengths.tolist()}")

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

    for i, m in enumerate(motif_length_range[::-1]):
        if m // subsample < data.shape[-1]:
            dist, candidates, candidate_dims, elbow_points, _, _ \
                = search_leitmotifs_elbow(
                k_max,
                data,
                m // subsample,
                n_dims=n_dims,
                n_jobs=n_jobs,
                elbow_deviation=elbow_deviation,
                minimize_pairwise_dist=minimize_pairwise_dist,
                slack=slack,
                distance=distance,
                distance_single=distance_single,
                distance_preprocessing=distance_preprocessing,
                backend=backend
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
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend='default',
        top_N=None,
):
    """Search leitmotifs for all ``k`` and return the elbow-function result.

    This is the functional API behind ``LAMA.fit_k_elbow``. It computes
    nearest-neighbor candidates, selects subdimensions when requested, evaluates
    leitmotif extents for ``k`` in ``[2, k_max]``, and detects elbow points from
    the rank-0 extent curve.

    Parameters
    ----------
    k_max : int
        Largest motif-set size to test.
    data : array-like
        Time series with dimensions on rows.
    motif_length : int
        Subsequence length.
    n_dims : int, optional
        Number of dimensions to use. If omitted, all dimensions are used.
    top_N : int, optional
        Number of ranked, non-overlapping leitmotifs to return per ``k``.
        ``None`` is equivalent to ``1``.

    Returns
    -------
    tuple
        By default returns ``(dists, candidates, candidate_dims, elbow_points,
        motif_length, memory_usage)``. With ``return_distances=True``, the
        motif length item is replaced by the reusable distance and k-NN state.
        For ``top_N > 1``, ``dists`` is two-dimensional and candidates/dims
        store rank arrays at each ``k``.
    """
    _, data_raw = pd_series_to_numpy(data)
    if motif_length <= 0 or motif_length >= data_raw.shape[-1]:
        raise ValueError(
            "motif_length must be positive and smaller than the time "
            f"series length ({data_raw.shape[-1]}). Got {motif_length}.")
    if top_N is None:
        top_N = 1
    elif not isinstance(top_N, (int, np.integer)) or top_N < 1:
        raise ValueError("top_N must be a positive integer or None.")

    n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
    previous_jobs = get_num_threads()
    set_num_threads(n_jobs)
    try:
        return _search_leitmotifs_elbow_impl(
            k_max,
            data_raw,
            motif_length,
            n_dims=n_dims,
            elbow_deviation=elbow_deviation,
            filter=filter,
            slack=slack,
            return_distances=return_distances,
            D_full=D_full,
            knns=knns,
            minimize_pairwise_dist=minimize_pairwise_dist,
            n_jobs=n_jobs,
            distance=distance,
            distance_single=distance_single,
            distance_preprocessing=distance_preprocessing,
            backend=backend,
            top_N=top_N)
    finally:
        set_num_threads(previous_jobs)


def _search_leitmotifs_elbow_impl(
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
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend='default',
        top_N=1,
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
    backend : String, default="scalable"
        The backend to use. As of now 'scalable' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory.
    top_N : int, default=1
        Number of ranked, non-overlapping leitmotifs to retain per ``k``.
        Elbow detection is always based on rank 0.

    Returns
    -------
    Tuple
        dists :
            Distances for each k in [2...k_max]. One-dimensional for
            ``top_N == 1`` and shaped ``(k, rank)`` otherwise.
        candidates :
            Leitmotif position candidates for each k. For ``top_N > 1``, each
            entry is shaped ``(top_N, k)``.
        candidate_dims :
            Selected dimensions matching each candidate. For ``top_N > 1``,
            each entry is shaped ``(top_N, n_dims_used)``.
        elbow_points :
            Elbow points selected from the rank-0 extent curve.
        m : int
            Motif length.
    """
    n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
    previous_jobs = get_num_threads()
    set_num_threads(n_jobs)

    # convert to numpy array
    _, data_raw = pd_series_to_numpy(data)

    # used memory
    process = psutil.Process()

    # m: motif size, n: number of subsequences, d: dimensions
    m = motif_length
    n = data_raw.shape[-1] - m + 1
    d = data_raw.shape[0]

    k_max_ = max(3, min(int(n // (m * slack)), k_max))

    # Check if use_dim is smaller than all given dimensions
    n_dims = d if n_dims is None else n_dims
    sum_dims = True if n_dims >= d else False

    scalable_gb = ((n ** 2) * d) * 32 / (1024 ** 3) / 8.0
    recommend_scalable = (scalable_gb > 8.0)

    if recommend_scalable and backend == "default":
        print(f"Setting 'scalable' backend for distance computations due to "
              f"excessive memory requirements. Old Backend: '{backend}'")
        backend = "scalable"
        recommend_scalable = False

    # order dimensions by increasing distance
    use_dim = min(n_dims, d)  # dimensions indexed by 0

    # compute the distance matrix
    if D_full is None:
        if minimize_pairwise_dist:  # FIXME: find better name
            # this has the drawback, that each pair of subsequences may
            # have different smallest dimensions

            print("Sort along dimension axis", flush=True)
            D_full, _ = compute_distances_with_knns_full(
                data_raw, m, k_max_,
                compute_knns=False,
                n_jobs=n_jobs,
                slack=slack,
                sum_dims=False,
                distance=distance,
                distance_single=distance_single,
                distance_preprocessing=distance_preprocessing,
            )

            D_full = np.sort(D_full, axis=0)[:n_dims].sum(axis=0, dtype=np.float32)
            knns = _argknns(D_full, k_max_, m, n, slack)

            D_full = D_full.reshape(1, D_full.shape[0], D_full.shape[1])
            knns = knns.reshape(1, knns.shape[0], knns.shape[1])
        elif backend == "scalable":
            D_knns, knns = compute_distances_with_knns(
                data_raw, m, k_max_,
                n_jobs=n_jobs,
                slack=slack,
                distance=distance,
                distance_single=distance_single,
                distance_preprocessing=distance_preprocessing
            )
            D_full = D_knns
        elif backend == 'default':
            D_full, knns = compute_distances_with_knns_full(
                data_raw, m, k_max_,
                n_jobs=n_jobs,
                slack=slack,
                sum_dims=sum_dims,
                distance=distance,
                distance_single=distance_single,
                distance_preprocessing=distance_preprocessing
            )
        else:
            raise ValueError(
                "No valid backend (combination) chosen. "
                "Please choose 'scalable' or 'default'.")

    print(f"Using '{backend}' Backend", flush=True)
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    # non-overlapping motifs only
    k_leitmotif_distances = np.full((k_max_ + 1, top_N), np.inf, dtype=np.float64)
    k_leitmotif_candidates = np.empty(k_max_ + 1, dtype=object)
    k_leitmotif_dims = np.empty(k_max_ + 1, dtype=object)

    upper_bound = np.inf
    preprocessing = []
    for dim in range(len(data_raw)):
        preprocessing.append(distance_preprocessing(data_raw[dim], m))
    preprocessing = np.array(preprocessing, dtype=np.float64)

    for test_k in range(k_max_, 1, -1):
        if minimize_pairwise_dist or sum_dims:
            # Do nothing
            dim_index = np.zeros((n, 1), dtype=np.int32)
        elif not sum_dims:
            # k-th NN and it's distance along all dimensions
            knn_idx = knns[:, :, test_k - 1]
            if backend == "scalable":
                D_knn = take_along_axis(D_knns, d, test_k - 1, n)
            else:
                D_knn = np.take_along_axis(
                    D_full,
                    knn_idx.reshape((knn_idx.shape[0], knn_idx.shape[1], 1)),
                    axis=2)[:, :, 0]

            dim_index = np.argsort(D_knn, axis=0)[:use_dim]
            dim_index = np.transpose(dim_index, (1, 0))
            dim_index = np.asarray(dim_index, dtype=np.int32)

        else:
            raise ValueError(
                "No valid backend (combination) chosen. "
                "Please choose 'scalable' or 'default'.")

        candidates, candidate_dists, candidate_dims = run_LAMA_top_n(
            data_raw, m, test_k, D_full, knns, dim_index,
            distance_single=distance_single,
            preprocessing=preprocessing,
            use_D_full=(backend != "scalable"),
            upper_bound=upper_bound,
            top_N=top_N,
        )

        k_leitmotif_distances[test_k, :len(candidate_dists)] = candidate_dists

        if minimize_pairwise_dist or sum_dims:
            all_dims = np.empty((top_N, d), dtype=np.int32)
            for rank in range(top_N):
                all_dims[rank] = np.arange(d, dtype=np.int32)
            candidate_dims = all_dims

        if top_N == 1:
            if candidates[0, 0] >= 0:
                k_leitmotif_candidates[test_k] = candidates[0]
                k_leitmotif_dims[test_k] = candidate_dims[0]
            else:
                k_leitmotif_candidates[test_k] = None
                k_leitmotif_dims[test_k] = None
        else:
            k_leitmotif_candidates[test_k] = candidates
            k_leitmotif_dims[test_k] = candidate_dims

    # smoothen the line to make it monotonically increasing
    k_leitmotif_distances[0:2] = k_leitmotif_distances[2]
    for i in range(len(k_leitmotif_distances) - 1, 2, -1):
        k_leitmotif_distances[i - 1] = np.minimum(
            k_leitmotif_distances[i], k_leitmotif_distances[i - 1])

    rank_zero_distances = k_leitmotif_distances[:, 0]
    elbow_points = find_elbow_points(rank_zero_distances,
                                     elbow_deviation=elbow_deviation)

    if filter:
        if top_N == 1:
            elbow_points = _filter_unique(
                elbow_points, k_leitmotif_candidates, motif_length)
        else:
            rank_zero_candidates = np.empty(len(k_leitmotif_candidates), dtype=object)
            for i in range(len(k_leitmotif_candidates)):
                if k_leitmotif_candidates[i] is not None:
                    k_candidates = k_leitmotif_candidates[i]
                    if k_candidates[0, 0] >= 0:
                        rank_zero_candidates[i] = k_candidates[0]
            elbow_points = _filter_unique(
                elbow_points, rank_zero_candidates, motif_length)

    set_num_threads(previous_jobs)

    # Cleanup
    if 'D_knns' in locals():
        del D_knns
    if 'D_knn' in locals():
        del D_knn

    if top_N == 1:
        k_leitmotif_distances_out = k_leitmotif_distances[:, 0]
    else:
        k_leitmotif_distances_out = k_leitmotif_distances

    if return_distances:
        return (k_leitmotif_distances_out, k_leitmotif_candidates, k_leitmotif_dims,
                elbow_points, D_full, knns, memory_usage)
    else:
        return (k_leitmotif_distances_out, k_leitmotif_candidates, k_leitmotif_dims,
                elbow_points, m, memory_usage)


@njit(cache=True)
def _argknns(D_full, k_max_, m, n, slack):
    # compute knns from new distance matrix
    knns = np.full((n, k_max_), -1, dtype=np.int32)
    for order in range(0, D_full.shape[0]):
        knn = _argknn(D_full[order], k_max_, m, slack=slack)
        knns[order, :len(knn)] = knn

    return knns


@njit(cache=True, parallel=True)
def take_along_axis(D_knns, d, knn, n):
    D_knn = np.zeros((d, n), dtype=np.float32)
    for dim in prange(d):
        for j in prange(n):
            D_knn[dim, j] = D_knns[dim][j][knn]
    return D_knn


@njit(cache=True)
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
    return compute_distances_with_knns_full(
        ts,
        m,
        1,
        exclude_trivial_match=exclude_trivial_match,
        n_jobs=n_jobs,
        slack=slack,
        sum_dims=True)[0]
