# -*- coding: utf-8 -*-
"""Plotting utilities.
"""

__author__ = ["patrickzib"]

import time

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore

import motiflets.motiflets as ml

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class Motiflets:

    def __init__(
            self,
            ds_name,
            series,
            ground_truth=None,
            dimension_labels=None,
            elbow_deviation=1.00,
            n_dims=None,
            n_jobs=4,
            slack=0.5,
    ):
        self.ds_name = ds_name
        self.series = convert_to_2d(series)

        self.elbow_deviation = elbow_deviation
        self.slack = slack
        self.dimension_labels = dimension_labels
        self.ground_truth = ground_truth

        self.motif_length_range = None
        self.motif_length = 0
        self.all_extrema = []
        self.all_elbows = []
        self.all_top_motiflets = []
        self.all_dists = []

        self.n_dims = n_dims
        self.n_jobs = n_jobs
        self.motif_length = 0
        self.k_max = 0
        self.dists = []
        self.motiflets = []
        self.elbow_points = []
        self.motiflets_dims = []
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
         self.motiflets,
         self.motiflets_dims,
         self.elbow_points,
         self.all_elbows,
         self.all_top_motiflets,
         self.all_dists,
         self.all_dimensions,
         self.all_extrema) = plot_motif_length_selection(
            k_max,
            self.series,
            motif_length_range,
            self.ds_name,
            n_dims=self.n_dims,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            subsample=subsample,
            plot_elbows=plot_elbows,
            plot_motifs=plot_motifsets,
            plot=plot,
            plot_best_only=plot_best_only)

        return self.motif_length, self.all_extrema

    def fit_k_elbow(
            self,
            k_max,
            motif_length=None,  # if None, use best_motif_length
            filter_duplicates=True,
            plot_elbows=True,
            plot_motifs_as_grid=True,
    ):
        self.k_max = k_max

        if motif_length is None:
            motif_length = self.motif_length
        else:
            self.motif_length = motif_length

        self.dists, self.motiflets, self.motiflets_dims, self.elbow_points = plot_elbow(
            k_max,
            self.series,
            n_dims=self.n_dims,
            ds_name=self.ds_name,
            motif_length=motif_length,
            plot_elbows=plot_elbows,
            plot_grid=plot_motifs_as_grid,
            ground_truth=self.ground_truth,
            dimension_labels=self.dimension_labels,
            filter=filter_duplicates,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack
        )

        return self.dists, self.motiflets, self.elbow_points

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
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
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

    def plot_motifset(self, elbow_points=None, path=None):

        if self.dists is None or self.motiflets is None or self.elbow_points is None:
            raise Exception("Please call fit_k_elbow first.")

        if elbow_points is None:
            elbow_points = self.elbow_points

        # TODO
        # if elbow_point is None:
        #    elbow_point = self.elbow_points[-1]

        fig, ax = plot_motifsets(
            self.ds_name,
            self.series,
            motifsets=self.motiflets[elbow_points],
            motiflet_dims=self.motiflets_dims[elbow_points],
            dist=self.dists[elbow_points],
            motif_length=self.motif_length,
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax


def convert_to_2d(series):
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

def as_series(data, index_range, index_name):
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
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show: boolean
        Outputs the plot

    """
    return plot_motifsets(ds_name, data, ground_truth=ground_truth, show=show)


def plot_motifsets(
        ds_name,
        data,
        motifsets=None,
        motifset_names=None,
        dist=None,
        motiflet_dims=None,
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
    motifsets: array like
        Found motif sets
    dist: array like
        The distances (extents) for each motif set
    motif_length: int
        The length of the motif
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show: boolean
        Outputs the plot

    """
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
                                     10 + 2 * len(motifsets), 5 + data.shape[0] // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': git_ratio,
                                     'height_ratios': [10, 1]})
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False,
                                 figsize=(20, 3 + data.shape[0] // 3))

    if ground_truth is None:
        ground_truth = []

    data_index, data_raw = ml.pd_series_to_numpy(data)

    offset = 0
    tick_offsets = []
    axes[0, 0].set_title(ds_name, fontsize=20)

    for dim in range(data_raw.shape[0]):
        dim_data_raw = zscore(data_raw[dim])
        offset -= 1.2 * (np.max(dim_data_raw) - np.min(dim_data_raw))
        tick_offsets.append(offset)

        _ = sns.lineplot(x=data_index,
                         y=dim_data_raw + offset,
                         ax=axes[0, 0],
                         linewidth=1,
                         color=sns.color_palette("tab10")[0],
                         ci=None,
                         estimator=None
                         )
        sns.despine()

        if motifsets is not None:
            for i, motifset in enumerate(motifsets):
                if (motiflet_dims is None or
                        (motiflet_dims[i] is not None and dim in motiflet_dims[i])):
                    if motifset is not None:
                        for a, pos in enumerate(motifset):
                            _ = sns.lineplot(ax=axes[0, 0],
                                             x=data_index[
                                                 np.arange(pos, pos + motif_length)],
                                             y=dim_data_raw[
                                               pos:pos + motif_length] + offset,
                                             linewidth=2,
                                             color=sns.color_palette("tab10")[2 + i],
                                             ci=None,
                                             estimator=None)

                            axes[0, 1 + i].set_title(
                                (("Motif Set " + str(i + 1)) if motifset_names is None
                                 else motifset_names[i]) + "\n" +
                                "k=" + str(len(motifset)) +
                                # ", d=" + str(np.round(dist[i], 2)) +
                                ", l=" + str(motif_length),
                                fontsize=16)

                            df = pd.DataFrame()
                            df["time"] = range(0, motif_length)

                            for aa, pos in enumerate(motifset):
                                df[str(aa)] = zscore(
                                    dim_data_raw[pos:pos + motif_length]) + offset

                            df_melt = pd.melt(df, id_vars="time")
                            _ = sns.lineplot(ax=axes[0, 1 + i],
                                             data=df_melt,
                                             ci=99,
                                             n_boot=10,
                                             lw=1,
                                             color=sns.color_palette("tab10")[2 + i],
                                             x="time",
                                             y="value")

        for aaa, column in enumerate(ground_truth):
            for offsets in ground_truth[column]:
                for pos, off in enumerate(offsets):
                    if pos == 0:
                        sns.lineplot(x=data_index[off[0]: off[1]],
                                     y=dim_data_raw[off[0]:off[1]] + offset,
                                     label=column,
                                     color=sns.color_palette("tab10")[(aaa + 1) % 10],
                                     ax=axes[0, 0],
                                     ci=None, estimator=None
                                     )
                    else:
                        sns.lineplot(x=data_index[off[0]: off[1]],
                                     y=dim_data_raw[off[0]:off[1]] + offset,
                                     color=sns.color_palette("tab10")[(aaa + 1) % 10],
                                     ax=axes[0, 0],
                                     ci=None, estimator=None
                                     )

    if motifsets is not None:
        y_labels = []
        for i, motiflet in enumerate(motifsets):
            if motiflet is not None:
                for aa, pos in enumerate(motiflet):
                    ratio = 0.8
                    rect = Rectangle(
                        (data_index[pos], -i),
                        data_index[pos + motif_length - 1] - data_index[pos],
                        ratio,
                        facecolor=sns.color_palette("tab10")[2 + i],
                        alpha=0.7
                    )
                    axes[1, 0].add_patch(rect)

                y_labels.append("Motif Set" + str(i))

        axes[1, 0].set_yticks(-np.arange(len(motifsets)) + 1.5)
        axes[1, 0].set_yticklabels([], fontsize=12)
        axes[1, 0].set_ylim([-abs(len(motifsets)) + 1, 1])
        axes[1, 0].set_xlim(axes[0, 0].get_xlim())
        axes[1, 0].set_title("Position of Motifsets", fontsize=20)

        for i in range(1, axes.shape[-1]):
            axes[1, i].remove()

    if isinstance(data, pd.DataFrame):
        axes[0, 0].set_yticks(tick_offsets)
        axes[0, 0].set_yticklabels(data.index, fontsize=12)

        if motifsets is not None:
            axes[0, 1].set_yticks(tick_offsets)
            axes[0, 1].set_yticklabels(data.index, fontsize=12)

    sns.despine()
    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def _plot_elbow_points(
        ds_name, data, motif_length,
        elbow_points,
        motifset_candidates,
        dists,
        motifset_candidates_dims=None):
    """Plots the elbow points found.

    Parameters
    ----------
    ds_name: String
        The name of the time series.
    data: array-like
        The time series data.
    motif_length: int
        The length of the motif.
    elbow_points: array-like
        The elbow points to plot.
    motifset_candidates: 2d array-like
        The motifset candidates. Will only extract those motif sets
        within elbow_points.
    dists: array-like
        The distances (extents) for each motif set
    """

    data_index, data_raw = ml.pd_series_to_numpy(data)

    # turn into 2d array
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))

    fig, ax = plt.subplots(figsize=(10, 4),
                           constrained_layout=True)
    ax.set_title(ds_name + "\nElbow Points")
    ax.plot(range(2, len(np.sqrt(dists))), dists[2:], "b", label="Extent")

    lim1 = plt.ylim()[0]
    lim2 = plt.ylim()[1]
    for elbow in elbow_points:
        ax.vlines(
            elbow, lim1, lim2,
            linestyles="--", label=str(elbow) + "-Motiflet"
        )
    ax.set(xlabel='Size (k)', ylabel='Extent')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.scatter(elbow_points, dists[elbow_points], color="red", label="Minima")

    motiflets = motifset_candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        if motiflet is not None:
            if elbow_points[i] - 3 < 0:
                x_pos = 0
            else:
                x_pos = (elbow_points[i] - 2) / (len(motifset_candidates))

            scale = max(dists) - min(dists)
            # y_pos = (dists[elbow_points[i]] - min(dists) + scale * 0.15) / scale
            axins = ax.inset_axes(
                [x_pos, 0.1, 0.2, 0.15])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for dim in range(data_raw.shape[0]):
                if (motifset_candidates_dims is None or
                        dim == motifset_candidates_dims[elbow_points][0][0]):
                    pos = motiflet[0]
                    normed_data = zscore(data_raw[dim, pos:pos + motif_length])
                    df["dim_" + str(dim)] = normed_data

            df_melt = pd.melt(df, id_vars="time")
            _ = sns.lineplot(ax=axins, data=df_melt,
                             x="time", y="value",
                             hue="variable",
                             style="variable",
                             ci=99,
                             # alpha=0.8,
                             n_boot=10, color=sns.color_palette("tab10")[(i + 1) % 10])
            axins.set_xlabel("")
            axins.patch.set_alpha(0)
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())
            axins.legend().set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_elbow(k_max,
               data,
               ds_name,
               motif_length,
               n_dims=2,
               plot_elbows=False,
               plot_grid=True,
               ground_truth=None,
               dimension_labels=None,
               filter=True,
               n_jobs=4,
               elbow_deviation=1.00,
               slack=0.5):
    """Plots the elbow-plot for k-Motiflets.

    This is the method to find and plot the characteristic k-Motiflets within range
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
    n_dims : int
        the number of dimensions to use for subdimensional motif discovery
    plot_elbows: bool, default=False
        plots the elbow ploints into the plot
    plot_grid: bool, default=True
        The motifs along the time series
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    dimension_labels:
        Labels for the dimensions
    filter: bool, default=True
        filters overlapping motiflets from the result,
    n_jobs : int
        Number of jobs to be used.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack : float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

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
    print("Data", raw_data.shape)

    startTime = time.perf_counter()
    dists, candidates, candidate_dims, elbow_points, m = ml.search_k_motiflets_elbow(
        k_max,
        raw_data,
        motif_length,
        n_dims=n_dims,
        elbow_deviation=elbow_deviation,
        filter=filter,
        n_jobs=n_jobs,
        slack=slack)
    endTime = (time.perf_counter() - startTime)

    print("Chosen window-size:", m, "in", np.round(endTime, 1), "s")
    print("Elbow Points", elbow_points)

    if plot_elbows:
        _plot_elbow_points(
            ds_name, data, motif_length, elbow_points,
            candidates, dists, motifset_candidates_dims=candidate_dims)

    if plot_grid:
        plot_motifsets(
            ds_name,
            data,
            motifsets=candidates[elbow_points],
            motiflet_dims=candidate_dims[elbow_points],
            dist=dists,
            motif_length=motif_length,
            show=True)

        # plot_grid_motiflets(
        #    ds_name, data, candidates, elbow_points,
        #    dists, motif_length, show_elbows=False,
        #    candidates_dims=candidate_dims,
        #    font_size=24,
        #    ground_truth=ground_truth,
        #    dimension_labels=dimension_labels)

    return dists, candidates, candidate_dims, elbow_points


def plot_motif_length_selection(
        k_max, data, motif_length_range, ds_name,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        subsample=2,
        n_dims=2,
        plot=True,
        plot_best_only=True,
        plot_elbows=True,
        plot_motifs=True,
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
    n_jobs : int
        Number of jobs to be used.
    elbow_deviation: float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack : float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

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
     elbow, top_motiflets,
     top_motiflets_dims, dists) = \
        ml.find_au_ef_motif_length(
            data_raw, k_max,
            n_dims=n_dims,
            motif_length_range=motif_length_range,
            n_jobs=n_jobs,
            elbow_deviation=elbow_deviation,
            slack=slack,
            subsample=subsample)
    endTime = (time.perf_counter() - startTime)
    print("\tTime", np.round(endTime, 1), "s")

    all_minima = _filter_duplicate_window_sizes(au_ef, all_minima)

    if plot:
        _plot_window_lengths(
            all_minima, au_ef, data_raw, ds_name, elbow, header, index,
            motif_length_range, top_motiflets,
            top_motiflets_dims=top_motiflets_dims)

        if plot_elbows or plot_motifs:
            to_plot = all_minima[0]
            if plot_best_only:
                to_plot = [np.argmin(au_ef)]

            for a in to_plot:
                motif_length = motif_length_range[a]
                candidates = np.zeros(len(dists[a]), dtype=object)
                candidates[elbow[a]] = top_motiflets[a]  # need to unpack

                candidate_dims = np.zeros(len(dists[a]), dtype=object)
                candidate_dims[elbow[a]] = top_motiflets_dims[a]  # need to unpack

                elbow_points = elbow[a]

                if plot_elbows:
                    _plot_elbow_points(
                        ds_name, data, motif_length,
                        elbow_points, candidates, dists[a],
                        motifset_candidates_dims=candidate_dims)

                if plot_motifs:
                    plot_motifsets(
                        ds_name,
                        data,
                        motifsets=top_motiflets[a],
                        motiflet_dims=top_motiflets_dims[a],
                        dist=dists[a][elbow_points],
                        motif_length=motif_length,
                        show=True)

    best_pos = np.argmin(au_ef)
    best_elbows = elbow[best_pos]
    best_dist = dists[best_pos]
    best_motiflets = np.zeros(len(dists[best_pos]), dtype=object)
    best_motiflets[elbow[best_pos]] = top_motiflets[best_pos]  # need to unpack
    best_motiflets_dims = np.zeros(len(dists[best_pos]), dtype=object)
    best_motiflets_dims[elbow[best_pos]] = top_motiflets_dims[
        best_pos]  # need to unpack

    return (best_motif_length,
            best_dist,
            best_motiflets,
            best_motiflets_dims,
            best_elbows,
            elbow,
            top_motiflets,
            dists,
            top_motiflets_dims,
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
        top_motiflets,
        top_motiflets_dims=None,
        font_size=20):
    set_sns_style(font_size)

    indices = ~np.isinf(au_ef)
    fig, ax = plt.subplots(figsize=(10, 3),
                           constrained_layout=True
                           )
    sns.lineplot(
        # x=index[motif_length_range[indices]],  # TODO!!!
        x=motif_length_range[indices],
        y=au_ef[indices],
        label="AU_EF",
        ci=None, estimator=None,
        ax=ax)
    sns.despine()
    ax.set_title("Best lengths on " + ds_name, size=20)
    ax.set(xlabel='Motif Length' + header, ylabel='Area under EF\n(lower is better)')
    ax.scatter(  # index[motif_length_range[all_minima]],   # TODO!!!
        motif_length_range[all_minima],
        au_ef[all_minima], color="red",
        label="Minima")
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    # turn into 2d array
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))
    # iterate all minima
    for i, minimum in enumerate(all_minima[0]):
        # iterate all motiflets
        for a, motiflet_pos in enumerate(top_motiflets[minimum]):
            x_pos = minimum / len(motif_length_range)
            scale = max(au_ef) - min(au_ef)
            y_pos = (au_ef[minimum] - min(au_ef) + (1.5 * a + 1) * scale * 0.15) / scale
            axins = ax.inset_axes([x_pos, y_pos, 0.20, 0.15])

            motif_length = motif_length_range[minimum]
            df = pd.DataFrame()
            df["time"] = index[range(0, motif_length)]

            for dim in range(data_raw.shape[0]):
                if top_motiflets_dims is None or dim == top_motiflets_dims[minimum][0][
                    0]:
                    pos = motiflet_pos[0]
                    normed_data = zscore(data_raw[dim, pos:pos + motif_length])
                    df["dim_" + str(dim)] = normed_data

            df_melt = pd.melt(df, id_vars="time")
            _ = sns.lineplot(ax=axins, data=df_melt,
                             x="time", y="value",
                             hue="variable",
                             style="variable",
                             ci=99,
                             n_boot=10,
                             lw=1,
                             color=sns.color_palette("tab10")[(i + 1) % 10])
            axins.set_xlabel("")
            axins.patch.set_alpha(0)
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())
            axins.legend().set_visible(False)
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.tight_layout()
    # plt.savefig("window_length.pdf")
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
