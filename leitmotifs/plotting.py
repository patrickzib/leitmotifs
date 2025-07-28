# -*- coding: utf-8 -*-
"""Plotting utilities.
"""

__author__ = ["patrickzib"]

import os
import time

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore
from tsdownsample import MinMaxLTTBDownsampler

import leitmotifs.lama as ml
from leitmotifs.distances import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class LAMA:
    """
        Class implementing the LAMA.

        Parameters
        ----------
        ds_name : str
            Name of the dataset.
        series : array-like
            The time series data.
        minimize_pairwise_dist: bool (default=False)
            If True, the pairwise distance is minimized. This is the mStamp-approach.
            It has the potential drawback, that each pair of subsequences may have
            different smallest dimensions.
        ground_truth : pd.Series (default=None)
            Ground-truth information as pd.Series.
        dimension_labels : array-like (default=None)
            Labels used for the dimensions' axis for plotting.
            If None, numeric indices are used.
        elbow_deviation : float, optional (default=1.00)
            The minimal absolute deviation needed to detect an elbow.
            It measures the absolute change in deviation from k to k+1.
            1.05 corresponds to 5% increase in deviation.
        n_dims : int, optional (default=None)
            The number of dimensions to be used in the subdimensional motif discovery.
            If none: all dimensions are used.
        distance: str (default="znormed_ed")
            The name of the distance function to be computed.
            Available options are:
                - 'znormed_ed' or 'znormed_euclidean' for z-normalized ED
                - 'ed' or 'euclidean' for the "normal" ED.
        n_jobs : int, optional (default=1)
            Amount of threads used in the k-nearest neighbour calculation.
        slack: float, optional (default=0.5)
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        backend : String, default="scalable"
            The backend to use. As of now 'scalable', 'sparse' and 'default' are supported.
            Use 'default' for the original exact implementation with excessive memory,
            Use 'scalable' for a scalable, exact implementation with less memory,
            Use 'sparse' for a scalable, exact implementation with more memory.

        Methods
        -------
        fit(time_series)
            Fit the KSN model to the input time series data.
        constrain(self, lbound, ubound)
            Return a constrained KSN model for the given temporal constraint.
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

        (self.motif_length,
         self.dists,
         self.leitmotifs,
         self.leitmotifs_dims,
         self.elbow_points,
         self.all_elbows,
         self.all_top_leitmotifs,
         self.all_dists,
         self.all_dimensions,
         self.all_extrema) = plot_motif_length_selection(
            k_max,
            self.series,
            motif_length_range,
            self.ds_name,
            n_dims=self.n_dims,
            minimize_pairwise_dist=self.minimize_pairwise_dist,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            subsample=subsample,
            plot_elbows=plot_elbows,
            plot_motif=plot_motifsets,
            ground_truth=self.ground_truth,
            plot=plot,
            plot_best_only=plot_best_only,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend
        )

        return self.motif_length, self.all_extrema

    def fit_k_elbow(
            self,
            k_max,
            motif_length=None,  # if None, use best_motif_length
            filter_duplicates=True,
            plot_elbows=True,
            plot_motifsets=True,
    ):
        self.k_max = k_max

        if motif_length is None:
            motif_length = self.motif_length
        else:
            self.motif_length = motif_length

        (self.dists, self.leitmotifs, self.leitmotifs_dims,
         self.elbow_points, self.memory_usage) = plot_elbow(
            k_max,
            self.series,
            n_dims=self.n_dims,
            ds_name=self.ds_name,
            motif_length=motif_length,
            plot_elbows=plot_elbows,
            plot_motif=plot_motifsets,
            ground_truth=self.ground_truth,
            dimension_labels=self.dimension_labels,
            filter=filter_duplicates,
            minimize_pairwise_dist=self.minimize_pairwise_dist,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend
        )

        return self.dists, self.leitmotifs, self.elbow_points

    def fit_dimensions(
            self,
            k_max,
            motif_length,
            dim_range
    ):

        all_dist, all_candidates, all_candidate_dims, all_elbow_points \
            = ml.select_subdimensions(
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
            distance_preprocessing=self.distance_preprocessing
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Dimension Plot")
        sns.lineplot(x=np.arange(1, 6, dtype=np.int32), y=all_dist, ax=ax)
        plt.tight_layout()
        plt.show()

    def plot_dataset(self, path=None):
        fig, ax = plot_dataset(
            self.ds_name,
            self.series,
            show=path is None,
            ground_truth=self.ground_truth)

        if path is not None:
            plt.savefig(path)
            plt.show()

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

        fig, ax = plot_motifsets(
            self.ds_name,
            self.series,
            motifsets=self.leitmotifs[elbow_points],
            leitmotif_dims=self.leitmotifs_dims[elbow_points],
            motifset_names=motifset_names,
            # dist=self.dists[elbow_points],
            ground_truth=self.ground_truth,
            motif_length=self.motif_length,
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax


def convert_to_2d(
        series
):
    if series.ndim == 1:
        print('Warning: The input dimension must be 2d.')
        if isinstance(series, pd.Series):
            series = series.to_frame().T
        elif isinstance(series, (np.ndarray, np.generic)):
            series = np.arange(series.shape[-1])
    if series.shape[0] > series.shape[1]:
        raise ('Warning: The input shape is wrong. Dimensions should be on rows. '
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


def plot_dataset(
        ds_name,
        data,
        ground_truth=None,
        show=True
):
    """Plots a time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series
    ground_truth: pd.Series (default=None)
        Ground-truth information as pd.Series.
    show: boolean (default=True)
        Outputs the plot

    """
    return plot_motifsets(ds_name, data, ground_truth=ground_truth, show=show)


def plot_motifsets(
        ds_name,
        data,
        motifsets=None,
        motifset_names=None,
        leitmotif_dims=None,
        motif_length=None,
        ground_truth=None,
        show=True):
    """Plots the data and the found motif sets.

    Parameters
    ----------
    ds_name: String,
        The name of the time series
    data: array-like
        The time series data
    motifsets: array like (default=None)
        Found motif sets
    dist: array like (default=None)
        The distances (extents) for each motif set
    motifset_names: array-like (default=None)
        The names of the motif sets
    leitmotif_dims: array-like (default=None)
        The dimensions of the leitmotifs
    motif_length: int (default=None)
        The length of the motif
    ground_truth: pd.Series (default=None)
        Ground-truth information as pd.Series.
    show: boolean (default=True)
        Outputs the plot

    """
    # set_sns_style(font_size)
    # sns.set(font_scale=3)
    sns.set(font="Calibri")
    sns.set_style("white")

    # turn into 2d array
    data = convert_to_2d(data)

    if motifsets is not None:
        git_ratio = [4]
        for _ in range(len(motifsets)):
            git_ratio.append(1)

        fig, axes = plt.subplots(2, 1 + len(motifsets),
                                 sharey="row",
                                 sharex=False,
                                 figsize=(
                                     10 + 2 * len(motifsets),
                                     5 + (data.shape[0] + len(motifsets)) // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': git_ratio,
                                     'height_ratios': [10, 3]})  # 5 for rolling stone?
    elif ground_truth is not None:
        fig, axes = plt.subplots(2, 1,
                                 sharey="row",
                                 sharex=False,
                                 figsize=(20, 5 + data.shape[0] // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': [4],
                                     'height_ratios': [10, 1]})
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False,
                                 figsize=(20, 5 + data.shape[0] // 2))

    if ground_truth is None:
        ground_truth = []

    data_index, data_raw = ml.pd_series_to_numpy(data)
    # data_raw_sampled, factor = ml._resample(data_raw, sampling_factor=500)
    # data_index_sampled, _ = ml._resample(data_index, sampling_factor=500)
    data_raw_sampled, data_index_sampled = data_raw, data_index

    factor = 1
    if data_raw.shape[-1] > 500:
        data_raw_sampled = np.zeros((data_raw.shape[0], 500))
        for i in range(data_raw.shape[0]):
            index = MinMaxLTTBDownsampler().downsample(
                np.ascontiguousarray(data_raw[i]), n_out=500)
            data_raw_sampled[i] = data_raw[i, index]

        data_index_sampled = data_index[index]
        factor = max(1, data_raw.shape[-1] / data_raw_sampled.shape[-1])
        if motifsets is not None:
            motifsets_sampled = list(map(lambda x: np.int32(x // factor), motifsets))

    color_offset = 1
    offset = 0
    tick_offsets = []
    axes[0, 0].set_title(ds_name, fontsize=22)

    for dim in range(data_raw.shape[0]):
        dim_raw = zscore(data_raw[dim])
        dim_raw_sampled = zscore(data_raw_sampled[dim])
        offset -= 1.2 * (np.max(dim_raw_sampled) - np.min(dim_raw_sampled))
        tick_offsets.append(offset)

        _ = sns.lineplot(x=data_index_sampled,
                         y=dim_raw_sampled + offset,
                         ax=axes[0, 0],
                         linewidth=0.5,
                         # color=sns.color_palette("tab10")[0],
                         color="gray",
                         errorbar=("ci", None),
                         estimator=None
                         )
        sns.despine()

        if motifsets is not None:
            for i, motifset in enumerate(motifsets_sampled):
                # TODO fixme/hack: pass actual motif length for SMM
                # if motifset_names[i] == "SMM":
                #   motif_length_sampled = max(4, 10 // factor)
                # else:
                motif_length_sampled = np.int32(max(2, motif_length // factor))

                if (leitmotif_dims is None or
                        (leitmotif_dims[i] is not None and dim in leitmotif_dims[i])):
                    if motifset is not None:
                        for a, pos in enumerate(motifset):
                            # Do not plot, if all dimensions are covered
                            if ((leitmotif_dims is None or
                                 leitmotif_dims[i].shape[0] < data_raw.shape[0])
                                    and (pos + motif_length_sampled <
                                         dim_raw_sampled.shape[0])):
                                _ = sns.lineplot(ax=axes[0, 0],
                                                 x=data_index_sampled[
                                                   pos: pos + motif_length_sampled],
                                                 y=dim_raw_sampled[
                                                   pos: pos + motif_length_sampled] + offset,
                                                 linewidth=3,
                                                 color=sns.color_palette("tab10")[
                                                     (color_offset + i) % len(
                                                         sns.color_palette("tab10"))],
                                                 errorbar=("ci", None),
                                                 # alpha=0.9,
                                                 estimator=None)

                            motif_length_disp = motif_length
                            # if motifset_names[i] == "SMM":
                            #   motif_length_disp = 10

                            axes[0, 1 + i].set_title(
                                (("Motif Set " + str(i + 1)) if motifset_names is None
                                 else motifset_names[i % len(motifset_names)]) + "\n" +
                                "k=" + str(len(motifset)) +
                                # ", d=" + str(np.round(dist[i], 2)) +
                                ", l=" + str(motif_length_disp),
                                fontsize=18)

                            df = pd.DataFrame()
                            df["time"] = range(0, motif_length_disp, 4)

                            for aa, pos in enumerate(motifsets[i]):
                                values = np.zeros(len(df["time"]), dtype=np.float32)
                                value = dim_raw[pos:pos + motif_length_disp:4]
                                values[:len(value)] = value

                                df[str(aa)] = (values - values.mean()) / (
                                        values.std() + 1e-4) + offset

                            df_melt = pd.melt(df, id_vars="time")
                            _ = sns.lineplot(
                                ax=axes[0, 1 + i],
                                data=df_melt,
                                errorbar=("ci", 99),
                                # err_style="band",
                                # estimator="median",
                                n_boot=1,
                                lw=1,
                                color=sns.color_palette("tab10")[
                                    (color_offset + i) % len(
                                        sns.color_palette("tab10"))],
                                x="time",
                                y="value")

    gt_count = 0
    y_labels = []
    motif_set_count = 0 if motifsets is None else len(motifsets)

    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for off in offsets:
                ratio = 0.8
                start = np.int32(off[0] // factor)
                end = np.int32(off[1] // factor)
                if end - 1 < dim_raw_sampled.shape[0]:
                    rect = Rectangle(
                        (data_index_sampled[start], 0),
                        data_index_sampled[end - 1] - data_index_sampled[start],
                        ratio,
                        facecolor=sns.color_palette("tab10")[
                            (color_offset + motif_set_count + aaa) %
                            len(sns.color_palette("tab10"))],
                        alpha=0.7
                    )

                    rx, ry = rect.get_xy()
                    cx = rx + rect.get_width() / 2.0
                    cy = ry + rect.get_height() / 2.0
                    axes[1, 0].annotate(column, (cx, cy),
                                        color='black',
                                        weight='bold',
                                        fontsize=12,
                                        ha='center',
                                        va='center')

                    axes[1, 0].add_patch(rect)

    if ground_truth is not None and len(ground_truth) > 0:
        gt_count = 1
        y_labels.append("Ground Truth")

    if motifsets is not None:
        for i, leitmotif in enumerate(motifsets_sampled):
            # if motifset_names[i] == "SMM":
            #    motif_length_sampled = max(4, 10 // factor)
            # else:
            motif_length_sampled = np.int32(max(2, motif_length // factor))

            if leitmotif is not None:
                for pos in leitmotif:
                    if pos + motif_length_sampled - 1 < dim_raw_sampled.shape[0]:
                        ratio = 0.8
                        rect = Rectangle(
                            (data_index_sampled[pos], -i - gt_count),
                            data_index_sampled[pos + motif_length_sampled - 1] -
                            data_index_sampled[pos],
                            ratio,
                            facecolor=sns.color_palette("tab10")[
                                (color_offset + i) % len(sns.color_palette("tab10"))],
                            alpha=0.7
                        )
                        axes[1, 0].add_patch(rect)

                label = (("Motif Set " + str(i + 1)) if motifset_names is None
                         else motifset_names[i % len(motifset_names)])
                y_labels.append(label)

    if len(y_labels) > 0:
        axes[1, 0].set_yticks(-np.arange(len(y_labels)) + 0.5)
        axes[1, 0].set_yticklabels(y_labels, fontsize=18)
        axes[1, 0].set_ylim([-abs(len(y_labels)) + 1, 1])
        axes[1, 0].set_xlim(axes[0, 0].get_xlim())
        axes[1, 0].set_xticklabels([])
        axes[1, 0].set_xticks([])

        if motifsets is not None:
            axes[1, 0].set_title("Positions", fontsize=22)

        for i in range(1, axes.shape[-1]):
            axes[1, i].remove()

    if isinstance(data, pd.DataFrame):
        axes[0, 0].set_yticks(tick_offsets)
        axes[0, 0].set_yticklabels(data.index, fontsize=18)
        axes[0, 0].set_xlabel("Time", fontsize=18)

        if motifsets is not None:
            axes[0, 1].set_yticks(tick_offsets)
            axes[0, 1].set_yticklabels(data.index, fontsize=18)
            axes[0, 1].set_xlabel("Length", fontsize=18)

    sns.despine()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def _plot_elbow_points(
        ds_name, data,
        elbow_points,
        motifset_candidates,
        dists):
    """Plots the elbow points found.

    Parameters
    ----------
    ds_name: String
        The name of the time series.
    data: array-like
        The time series data.
    elbow_points: array-like
        The elbow points to plot.
    motifset_candidates: 2d array-like
        The motifset candidates. Will only extract those motif sets
        within elbow_points.
    dists: array-like
        The distances (extents) for each motif set
    """

    # data_index, data_raw = ml.pd_series_to_numpy(data)
    # turn into 2d array
    # if data_raw.ndim == 1:
    #    data_raw = data_raw.reshape((1, -1))

    fig, ax = plt.subplots(figsize=(10, 4),
                           constrained_layout=True)
    ax.set_title(ds_name + "\nElbow Points")
    ax.plot(range(2, len(np.sqrt(dists))), dists[2:], "b", label="Extent")

    lim1 = plt.ylim()[0]
    lim2 = plt.ylim()[1]
    for elbow in elbow_points:
        ax.vlines(
            elbow, lim1, lim2,
            linestyles="--", label=str(elbow) + "-Leitmotif"
        )
    ax.set(xlabel='Size (k)', ylabel='Extent')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.scatter(elbow_points, dists[elbow_points], color="red", label="Minima")

    # leitmotifs = motifset_candidates[elbow_points]

    plt.tight_layout()
    # plt.savefig("lord_of_the_rings_elbow_points.pdf")
    plt.show()


def plot_elbow(
        k_max,
        data,
        ds_name,
        motif_length,
        n_dims=2,
        plot_elbows=False,
        plot_motif=True,
        ground_truth=None,
        dimension_labels=None,
        minimize_pairwise_dist=False,
        filter=True,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance,
        distance_preprocessing=sliding_mean_std,
        backend="default"
):
    """Plots the elbow-plot for leitmotifs.

    This is the method to find and plot the characteristic leitmotifs within range
    [2...k_max] for given a `motif_length` using elbow-plots.

    Details are given within the paper Section 5.1 Learning meaningful k.

    Parameters
    ----------
    k_max: int
        use [2...k_max] to compute the elbow plot (user parameter).
    data: array-like
        the TS
    ds_name: String
        the name of the dataset
    motif_length: int
        the length of the motif (user parameter)
    n_dims : int (default=2)
        the number of dimensions to use for sub-dimensional motif discovery
    plot_elbows: bool (default=False)
        plots the elbow points into the plot
    plot_motif: bool (default=True)
        The motifs along the time series
    ground_truth: pd.Series (default=None)
        Ground-truth information as pd.Series.
    dimension_labels: array-like (default=None)
        Labels for the dimensions
    minimize_pairwise_dist: bool (default=False)
        If True, the pairwise distance is minimized. This is the mStamp-approach.
        It has the potential drawback, that each pair of subsequences may have
        different smallest dimensions.
    filter: bool (default=True)
        filters overlapping leitmotifs from the result,
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
        The backend to use. As of now 'scalable', 'sparse' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory,
        Use 'sparse' for a scalable, exact implementation with more memory.
    Returns
    -------
    Tuple
        dists:          distances for each k in [2...k_max]
        candidates:     motifset-candidates for each k
        elbow_points:   elbow-points

    """
    # turn into 2d array
    if data.ndim == 1:
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        elif isinstance(data, (np.ndarray, np.generic)):
            data = np.arange(data.shape[-1])

    _, raw_data = ml.pd_series_to_numpy(data)
    # print("Data", raw_data.shape)

    startTime = time.perf_counter()
    dists, candidates, candidate_dims, elbow_points, m, memory_usage = (
        ml.search_leitmotifs_elbow(
            k_max,
            raw_data,
            motif_length,
            n_dims=n_dims,
            elbow_deviation=elbow_deviation,
            filter=filter,
            n_jobs=n_jobs,
            minimize_pairwise_dist=minimize_pairwise_dist,
            slack=slack,
            distance=distance,
            distance_single=distance_single,
            distance_preprocessing=distance_preprocessing,
            backend=backend
        )
    )
    endTime = (time.perf_counter() - startTime)

    # print("Window-size:", m)
    # print("Elbow Points", elbow_points, " found in", np.round(endTime, 1), "s")

    if plot_elbows:
        _plot_elbow_points(
            ds_name, data, elbow_points,
            candidates, dists)

    if plot_motif:
        plot_motifsets(
            ds_name,
            data,
            motifsets=candidates[elbow_points],
            leitmotif_dims=candidate_dims[elbow_points],
            motif_length=motif_length,
            ground_truth=ground_truth,
            show=True)

        # plot_grid_leitmotifs(
        #    ds_name, data, candidates, elbow_points,
        #    dists, motif_length, show_elbows=False,
        #    candidates_dims=candidate_dims,
        #    font_size=24,
        #    ground_truth=ground_truth,
        #    dimension_labels=dimension_labels)

    return dists, candidates, candidate_dims, elbow_points, memory_usage


def plot_motif_length_selection(
        k_max,
        data,
        motif_length_range,
        ds_name,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        subsample=2,
        n_dims=None,
        minimize_pairwise_dist=False,
        ground_truth=None,
        plot=True,
        plot_best_only=True,
        plot_elbows=True,
        plot_motif=True,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance,
        distance_preprocessing=sliding_mean_std,
        backend="default"
):
    """Computes the AU_EF plot to extract the best motif lengths

    This is the method to find and plot the characteristic motif-lengths, for k in
    [2...k_max], using the area AU-EF plot.

    Details are given within the paper 5.2 Learning Motif Length l.

    Parameters
    ----------
    k_max: int
        use [2...k_max] to compute the elbow plot.
    data: array-like
        the TS
    motif_length_range: array-like
        the interval of lengths
    ds_name: String
        Name of the time series for displaying
    n_jobs : int (default=4)
        Number of jobs to be used.
    elbow_deviation: float (default=1.00)
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack : float (default=0.5)
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    subsample: int (default=2)
        The subsampling factor for the time series.
    n_dims : int, optional (default=None)
        The number of dimensions to be used in the subdimensional motif discovery.
        If none: all dimensions are used.
    minimize_pairwise_dist: bool (default=False)
        If True, the pairwise distance is minimized. This is the mStamp-approach.
        It has the potential drawback, that each pair of subsequences may have
        different smallest dimensions.
    plot: bool (default=True)
        Enables or disables plotting
    plot_best_only: bool (default=True)
        If True, only the leitmotif for the best motif length is plotted.
    plot_elbows: bool (default=True)
        If True, the elbow points are plotted.
    plot_motif: bool (default=True)
        If True, the motif sets are plotted.
    distance: callable (default=znormed_euclidean_distance)
        The distance function to be computed.
    distance_preprocessing: callable (default=sliding_mean_std)
        The distance preprocessing function to be computed.
    backend : String, default="scalable"
        The backend to use. As of now 'scalable', 'sparse' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory,
        Use 'sparse' for a scalable, exact implementation with more memory.


    Returns
    -------
    best_motif_length: int
        The motif length that maximizes the AU-EF.

    all_minima: int
        The local minima of the AU_EF

    """
    # turn into 2d array
    data = convert_to_2d(data)
    index, data_raw = ml.pd_series_to_numpy(data)

    header = " in " + data.index.name if isinstance(
        data, pd.Series) and data.index.name != None else ""

    # discretizes ranges
    motif_length_range = np.int32(motif_length_range)

    startTime = time.perf_counter()
    (best_motif_length,
     all_minima, au_ef,
     elbow, top_leitmotifs,
     top_leitmotifs_dims, dists) = \
        ml.find_au_ef_motif_length(
            data_raw, k_max,
            n_dims=n_dims,
            motif_length_range=motif_length_range,
            minimize_pairwise_dist=minimize_pairwise_dist,
            n_jobs=n_jobs,
            elbow_deviation=elbow_deviation,
            slack=slack,
            subsample=subsample,
            distance=distance,
            distance_single=distance_single,
            distance_preprocessing=distance_preprocessing,
            backend=backend
        )
    endTime = (time.perf_counter() - startTime)
    # print("\tTime", np.round(endTime, 1), "s")

    all_minima = _filter_duplicate_window_sizes(au_ef, all_minima)

    if plot:
        _plot_window_lengths(
            all_minima, au_ef, data_raw, ds_name, elbow, header, index,
            motif_length_range, top_leitmotifs,
            top_leitmotifs_dims=top_leitmotifs_dims)

        if plot_elbows or plot_motifsets:
            to_plot = all_minima[0]
            if plot_best_only:
                to_plot = [np.argmin(au_ef)]

            for a in to_plot:
                motif_length = motif_length_range[a]
                candidates = np.zeros(len(dists[a]), dtype=object)
                candidates[elbow[a]] = top_leitmotifs[a]  # need to unpack

                candidate_dims = np.zeros(len(dists[a]), dtype=object)
                candidate_dims[elbow[a]] = top_leitmotifs_dims[a]  # need to unpack

                elbow_points = elbow[a]

                if plot_elbows:
                    _plot_elbow_points(
                        ds_name, data,
                        elbow_points, candidates, dists[a])

                if plot_motif:
                    plot_motifsets(
                        ds_name,
                        data,
                        motifsets=top_leitmotifs[a],
                        leitmotif_dims=top_leitmotifs_dims[a],
                        motif_length=motif_length,
                        ground_truth=ground_truth,
                        show=True)

    best_pos = np.argmin(au_ef)
    best_elbows = elbow[best_pos]
    best_dist = dists[best_pos]
    best_leitmotifs = np.zeros(len(dists[best_pos]), dtype=object)
    best_leitmotifs[elbow[best_pos]] = top_leitmotifs[best_pos]  # need to unpack
    best_leitmotifs_dims = np.zeros(len(dists[best_pos]), dtype=object)
    best_leitmotifs_dims[elbow[best_pos]] = top_leitmotifs_dims[
        best_pos]  # need to unpack

    return (best_motif_length,
            best_dist,
            best_leitmotifs,
            best_leitmotifs_dims,
            best_elbows,
            elbow,
            top_leitmotifs,
            dists,
            top_leitmotifs_dims,
            all_minima[0])


def _filter_duplicate_window_sizes(au_ef, minima):
    """Filter neighboring window sizes with equal minima
    """
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


def _plot_window_lengths(
        all_minima, au_ef, data_raw, ds_name,
        elbow, header, index,
        motif_length_range,
        top_leitmotifs,
        top_leitmotifs_dims=None):
    # set_sns_style(font_size)

    indices = ~np.isinf(au_ef)
    fig, ax = plt.subplots(figsize=(10, 4),
                           constrained_layout=True
                           )
    sns.lineplot(
        # x=index[motif_length_range[indices]],  # TODO!!!
        x=motif_length_range[indices],
        y=au_ef[indices],
        label="AU_EF",
        errorbar=("ci", None), estimator=None,
        ax=ax)
    sns.despine()
    ax.set_title("Best lengths on " + ds_name, size=14)
    ax.set(xlabel='Motif Length' + header, ylabel='Area under EF\n(lower is better)')
    ax.scatter(  # index[motif_length_range[all_minima]],   # TODO!!!
        motif_length_range[all_minima],
        au_ef[all_minima], color="red",
        label="Minima")
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    # turn into 2d array
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))
    # iterate all minima
    for i, minimum in enumerate(all_minima[0]):
        # iterate all leitmotifs
        for a, leitmotif_pos in enumerate(top_leitmotifs[minimum]):
            x_pos = minimum / len(motif_length_range)
            scale = max(au_ef) - min(au_ef)
            y_pos = (au_ef[minimum] - min(au_ef) + (1.5 * a + 1) * scale * 0.15) / scale
            axins = ax.inset_axes([x_pos, y_pos, 0.20, 0.15])

            motif_length = motif_length_range[minimum]
            df = pd.DataFrame()
            df["time"] = index[range(0, motif_length)]

            for dim in range(data_raw.shape[0]):
                if top_leitmotifs_dims is None or dim == \
                        top_leitmotifs_dims[minimum][0][
                            0]:
                    pos = leitmotif_pos[0]
                    normed_data = zscore(data_raw[dim, pos:pos + motif_length])
                    df["dim_" + str(dim)] = normed_data

            df_melt = pd.melt(df, id_vars="time")
            _ = sns.lineplot(ax=axins, data=df_melt,
                             x="time", y="value",
                             hue="variable",
                             style="variable",
                             errorbar=("ci", 99),
                             n_boot=1,
                             lw=1,
                             color=sns.color_palette("tab10")[(i + 1) % 10])
            axins.set_xlabel("")
            axins.patch.set_alpha(0)
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())
            axins.legend().set_visible(False)
    # fig.set_figheight(5)
    # fig.set_figwidth(8)
    plt.tight_layout()
    plt.savefig("lord_of_the_rings_window_length.pdf")
    plt.show()


def set_sns_style(font_size):
    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_context("paper",
                    rc={"font.size": font_size,
                        "axes.titlesize": font_size - 8,
                        "axes.labelsize": font_size - 8,
                        "xtick.labelsize": font_size - 10,
                        "ytick.labelsize": font_size - 10, })
