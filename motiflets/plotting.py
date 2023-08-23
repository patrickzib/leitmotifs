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
import scipy.cluster.hierarchy as sch

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
            slack=0.5,
    ):
        self.ds_name = ds_name
        self.series = series
        self.elbow_deviation = elbow_deviation
        self.slack = slack
        self.dimension_labels = dimension_labels
        self.ground_truth = ground_truth

        self.motif_length_range = None
        self.best_motif_length = 0
        self.all_extrema = []

        self.motif_length = 0
        self.k_max = 0
        self.dists = []
        self.motiflets = []
        self.elbow_points = []

    def fit_motif_length(
            self,
            k_max,
            motif_length_range,
            subsample=2,
            plot=True,
            plot_elbows=False,
            plot_motifs_as_grid=True
    ):

        self.motif_length_range = motif_length_range
        self.k_max = k_max

        self.best_motif_length, self.all_extrema = plot_motif_length_selection(
            k_max,
            self.series,
            motif_length_range,
            self.ds_name,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            subsample=subsample,
            plot_elbows=plot_elbows,
            plot_grid=plot_motifs_as_grid,
            plot=plot)

        return self.best_motif_length, self.all_extrema

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
            motif_length = self.best_motif_length
        else:
            self.motif_length = motif_length

        self.dists, self.motiflets, self.elbow_points = plot_elbow(
            k_max,
            self.series,
            ds_name=self.ds_name,
            motif_length=motif_length,
            plot_elbows=plot_elbows,
            plot_grid=plot_motifs_as_grid,
            ground_truth=self.ground_truth,
            dimension_labels=self.dimension_labels,
            filter=filter_duplicates,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack
        )

        return self.dists, self.motiflets, self.elbow_points

    def fit_dendrogram(
            self,
            k_max,
            motif_length,
            n_clusters
    ):

        dists, motiflets, elbow_points = plot_elbow_by_dimension(
            k_max, self.series,
            dimension_labels=self.dimension_labels,
            ds_name=self.ds_name,
            motif_length=motif_length,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack
        )

        series = np.zeros((self.series.shape[0], self.series.shape[1] - motif_length),
                          dtype=np.float32)
        for i in range(series.shape[0]):
            for pos in motiflets[i, elbow_points[i][-1]]:  #
                series[i, max(0, pos - motif_length):min(pos + 2 * motif_length,
                                                         series.shape[1])] = 1
        X = series

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        Z = sch.linkage(X, method='single')

        # creating the dendrogram
        _ = sch.dendrogram(Z, labels=self.dimension_labels, ax=ax)

        ax.set_title('Dendrogram')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Euclidean distances')
        plt.tight_layout()
        plt.show()

        cluster_k = n_clusters
        y_dimensions = sch.fcluster(Z, cluster_k, criterion='maxclust')
        mapping = list(zip(y_dimensions, self.dimension_labels))

        joint_clusters = {}
        for i in range(1, cluster_k + 1):
            print("Cluster", i)
            joint_clusters[i] = [x[1] for x in mapping if x[0] == i]
            print(joint_clusters[i])
            print("----")

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

    def plot_motifset(self, path=None):

        if self.dists is None or self.motiflets is None or self.elbow_points is None:
            raise Exception("Please call fit_k_elbow first.")

        fig, ax = plot_motifset(
            self.ds_name,
            self.series,
            motifset=self.motiflets[self.elbow_points[-1]],
            dist=self.dists[self.elbow_points[-1]],
            motif_length=self.motif_length,
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax


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
    return plot_motifset(ds_name, data, ground_truth=ground_truth, show=show)


def append_all_motif_sets(df, motif_sets, method_name, D_full):
    """Utility function.

    Parameters
    ----------
    df: pd.DataFrame
        a dataframe to append to
    motif_sets: 2d-array-like
        The motif-sets to append under row `method_name`
    method_name: String
        The column to append as
    D_full:
        The distance matrix

    Returns
    -------
    df: pd.DataFrame
        the dataframe with appended data

    """

    filtered_motif_sets = [m for m in motif_sets if m is not None]
    extent = [ml.get_pairwise_extent(D_full, motiflet) for motiflet in
              filtered_motif_sets]
    count = [len(motiflet) for motiflet in filtered_motif_sets]

    for m, e, c in zip(filtered_motif_sets, extent, count):
        entry = {"Method": method_name, "Motif": m, "Extent": e, "k": c}
        df = df.append(entry, ignore_index=True)
    return df


def plot_motifset(
        ds_name,
        data,
        motifset=None,
        dist=None,
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
    motifset: array like
        One found motif set
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
    if data.ndim != 2:
        raise ValueError('The input dimension must be 2d.')

    if motifset is not None:
        fig, axes = plt.subplots(1, 2, sharey=False,
                                 sharex=False,
                                 figsize=(15, 3 + data.shape[0]),
                                 gridspec_kw={'width_ratios': [4, 1]})
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        axes = [axes]

    if ground_truth is None:
        ground_truth = []

    data_index, data_raw = ml.pd_series_to_numpy(data)

    offset = 0
    tick_offsets = []
    axes[0].set_title(ds_name, fontsize=20)
    for dim in range(data_raw.shape[0]):
        dim_data_raw = zscore(data_raw[dim])
        offset -= (np.max(dim_data_raw) - np.min(dim_data_raw))
        tick_offsets.append(offset)

        _ = sns.lineplot(x=data_index,
                         y=dim_data_raw + offset,
                         ax=axes[0],
                         linewidth=1,
                         color=sns.color_palette("tab10")[0],
                         ci=None,
                         estimator=None
                         )
        sns.despine()

        if motifset is not None:
            for a, pos in enumerate(motifset):
                _ = sns.lineplot(ax=axes[0],
                                 x=data_index[np.arange(pos, pos + motif_length)],
                                 y=dim_data_raw[pos:pos + motif_length] + offset,
                                 linewidth=2,
                                 color=sns.color_palette("tab10")[
                                     # (1+a) % len(sns.color_palette("tab10"))
                                     1],
                                 # alpha=0.5,
                                 ci=None,
                                 estimator=None)

        for aaa, column in enumerate(ground_truth):
            for offsets in ground_truth[column]:
                for pos, off in enumerate(offsets):
                    if pos == 0:
                        sns.lineplot(x=data_index[off[0]: off[1]],
                                     y=dim_data_raw[off[0]:off[1]] + offset,
                                     label=column,
                                     color=sns.color_palette("tab10")[(aaa + 1) % 10],
                                     ax=axes[0],
                                     ci=None, estimator=None
                                     )
                    else:
                        sns.lineplot(x=data_index[off[0]: off[1]],
                                     y=dim_data_raw[off[0]:off[1]] + offset,
                                     color=sns.color_palette("tab10")[(aaa + 1) % 10],
                                     ax=axes[0],
                                     ci=None, estimator=None
                                     )

        if motifset is not None:
            axes[1].set_title(
                "Motif Set, k=" + str(len(motifset)) + ", d=" + str(np.round(dist, 2)),
                fontsize=20)

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for aa, pos in enumerate(motifset):
                df[str(aa)] = zscore(dim_data_raw[pos:pos + motif_length]) + offset

            df_melt = pd.melt(df, id_vars="time")
            _ = sns.lineplot(ax=axes[1],
                             data=df_melt,
                             ci=99, n_boot=10,
                             x="time",
                             y="value")

    if isinstance(data, pd.DataFrame):
        axes[0].set_yticks(tick_offsets)
        axes[0].set_yticklabels(data.index, fontsize=12)

        if motifset is not None:
            axes[1].set_yticks(tick_offsets)
            axes[1].set_yticklabels(data.index, fontsize=12)

    sns.despine()

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def _plot_elbow_points(
        ds_name, data, motif_length,
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
                [x_pos, 0.1, 0.1, 0.15])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for dim in range(data_raw.shape[0]):
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
                             n_boot=10, color=sns.color_palette("tab10")[i % 10])
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
               plot_elbows=False,
               plot_grid=True,
               ground_truth=None,
               dimension_labels=None,
               filter=True,
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
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.


    Returns
    -------
    Tuple
        dists:          distances for each k in [2...k_max]
        candidates:     motifset-candidates for each k
        elbow_points:   elbow-points

    """
    _, raw_data = ml.pd_series_to_numpy(data)
    print("Data", raw_data.shape)

    # turn into 2d array
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape((1, -1))

    startTime = time.perf_counter()
    dists, candidates, elbow_points, slopes, m = ml.search_k_motiflets_elbow(
        k_max,
        raw_data,
        motif_length,
        elbow_deviation=elbow_deviation,
        filter=filter,
        slack=slack)
    endTime = (time.perf_counter() - startTime)

    print("Chosen window-size:", m, "in", np.round(endTime, 1), "s")
    print("Elbow Points", elbow_points)

    if plot_elbows:
        _plot_elbow_points(ds_name, data, motif_length, elbow_points, candidates, dists)

    if plot_grid:
        plot_grid_motiflets(
            ds_name, data, candidates, elbow_points,
            dists, motif_length, show_elbows=False,
            font_size=24,
            ground_truth=ground_truth,
            dimension_labels=dimension_labels)

    return dists, candidates, elbow_points


def plot_elbow_by_dimension(k_max,
                            data,
                            ds_name,
                            motif_length,
                            ground_truth=None,
                            dimension_labels=None,
                            filter=True,
                            elbow_deviation=1.00,
                            slack=0.5):
    """Searches and plots elbows of the k-Motiflets by dimension.

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
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    dimension_labels: list
        List of labels for each dimension
    filter: bool, default=True
        filters overlapping motiflets from the result,
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    Tuple
        dists:          distances for each k in [2...k_max] and dimension
        candidates:     motifset-candidates for each k and dimension
        elbow_points:   elbow-points for each dimension

    """
    _, raw_data = ml.pd_series_to_numpy(data)
    print("Data", raw_data.shape)

    startTime = time.perf_counter()
    dists, candidates, elbow_points = ml.search_multidim_k_motiflets_elbow(
        k_max,
        raw_data,
        motif_length,
        elbow_deviation=elbow_deviation,
        filter=filter,
        slack=slack)
    endTime = (time.perf_counter() - startTime)

    print("Compute elbows in", np.round(endTime, 1), "s")

    plot_motiflets_by_dimension(
        ds_name, data, candidates, elbow_points,
        dists, motif_length,
        dimension_labels=dimension_labels,
        font_size=24,
        ground_truth=ground_truth)

    return dists, candidates, elbow_points


def plot_motif_length_selection(
        k_max, data, motif_length_range, ds_name,
        elbow_deviation=1.00, slack=0.5, subsample=2,
        ground_truth=None,
        dimension_labels=None,
        plot=True,
        plot_elbows=True,
        plot_grid=True,
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
    elbow_deviation: float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    dimension_labels: list
        List of labels for each dimension

    Returns
    -------
    best_motif_length: int
        The motif length that maximizes the AU-EF.

    all_minima: int
        The local minima of the AU_EF

    """
    index, data_raw = ml.pd_series_to_numpy(data)

    # turn into 2d array
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))

    header = " in " + data.index.name if isinstance(
        data, pd.Series) and data.index.name != None else ""

    # discretizes ranges
    motif_length_range = np.int32(motif_length_range)

    startTime = time.perf_counter()
    (best_motif_length,
     all_minima, au_ef,
     all_slope_maxima, slope_ef,
     elbow, top_motiflets, dists) = \
        ml.find_au_ef_motif_length(
            data_raw, k_max,
            motif_length_range=motif_length_range,
            elbow_deviation=elbow_deviation,
            slack=slack,
            subsample=subsample)
    endTime = (time.perf_counter() - startTime)
    print("\tTime", np.round(endTime, 1), "s")

    # Find unique motif lengths (filters neigboring minima)
    # TODO best_motif_length can be missing!
    # best = ml._filter_unique(
    #    np.arange(len(top_motiflets[all_minima])),
    #    slope_ef[all_minima],
    #    top_motiflets[all_minima],
    #    np.max(motif_length_range[all_minima])
    # )
    # all_minima = [all_minima[0][best]]
    all_minima = _filter_duplicate_window_sizes(au_ef, all_minima)

    if plot:
        _plot_window_lengths(
            all_minima, au_ef, data_raw, ds_name, elbow, header, index,
            motif_length_range, top_motiflets)

        #_plot_window_lengths(
        #    all_slope_maxima, slope_ef, data_raw, ds_name, elbow, header, index,
        #    motif_length_range, top_motiflets)

        for a in all_minima[0]:
            motif_length = motif_length_range[a]
            candidates = np.zeros(len(dists[a]), dtype=np.object)
            candidates[elbow[a]] = top_motiflets[a]  # need to unpack
            elbow_points = elbow[a]
            dist = dists[a]

            if plot_elbows:
                _plot_elbow_points(
                    ds_name, data, motif_length, elbow_points, candidates, dist)

            if plot_grid:
                plot_grid_motiflets(
                    ds_name, data, candidates, elbow_points,
                    dists, motif_length, show_elbows=False,
                    font_size=24,
                    ground_truth=ground_truth,
                    dimension_labels=dimension_labels)

    return best_motif_length, all_minima[0]  # all_minima[0]


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
        motif_length_range, top_motiflets):
    indices = ~np.isinf(au_ef)
    fig, ax = plt.subplots(figsize=(10, 3),
                           constrained_layout=True)
    ax = sns.lineplot(
        x=index[motif_length_range[indices]],
        y=au_ef[indices],
        label="AU_EF",
        ci=None, estimator=None)
    sns.despine()
    ax.set_title("Best lengths on " + ds_name, size=20)
    ax.set(xlabel='Motif Length' + header, ylabel='Area under EF\n(lower is better)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.scatter(index[motif_length_range[all_minima]], au_ef[all_minima], color="red",
                label="Minima")
    print("Minima", index[motif_length_range[all_minima]], "Elbows", elbow[all_minima])
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    # turn into 2d array
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))
    # iterate all minima
    if False:
        for i, minimum in enumerate(all_minima[0]):
            # iterate all motiflets
            for a, motiflet_pos in enumerate(top_motiflets[minimum]):
                x_pos = minimum / len(motif_length_range)
                scale = max(au_ef) - min(au_ef)
                y_pos = (au_ef[minimum] - min(au_ef) + (1.5*a+1) * scale * 0.15) / scale
                axins = ax.inset_axes([x_pos, y_pos, 0.10, 0.15])

                motif_length = motif_length_range[minimum]
                df = pd.DataFrame()
                df["time"] = index[range(0, motif_length)]

                for j, dim in enumerate(range(data_raw.shape[0])):
                    pos = motiflet_pos[0]
                    normed_data = zscore(data_raw[dim, pos:pos + motif_length])
                    df["dim_" + str(dim)] = normed_data  # - 2 * j

                df_melt = pd.melt(df, id_vars="time")
                _ = sns.lineplot(ax=axins, data=df_melt,
                                 x="time", y="value",
                                 hue="variable",
                                 style="variable",
                                 ci=99,
                                 # alpha=0.8,
                                 n_boot=10, color=sns.color_palette("tab10")[i % 10])
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


def plot_motiflets_by_dimension(
        ds_name, data, candidates, elbow_points, dist,
        motif_length, font_size=20,
        ground_truth=None, dimension_labels=None,
        color_palette=sns.color_palette("tab10")
):
    """Plots the characteristic motifs found by dimension along the time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series data
    candidates: 2d array-like
        The motifset candidates
    elbow_points: array-like
        The elbow points found. Only motif sets from the elbow points will be plotted.
    dist: array-like
        The distances (extents) of the motif set candidates
    motif_length: int
        The motif length found.
    font_size: int
        Font-size to use for plotting.
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    dimension_labels:
        Labels for the dimensions
    color_palette:
        Color-palette to use
    """

    # turn into 2d array
    if data.ndim != 2:
        raise ValueError('The input dimension must be 2d.')

    data_index, data_raw = ml.pd_series_to_numpy(data)

    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_context("paper",
                    rc={"font.size": font_size,
                        "axes.titlesize": font_size - 8,
                        "axes.labelsize": font_size - 8,
                        "xtick.labelsize": font_size - 10,
                        "ytick.labelsize": font_size - 10, })

    if ground_truth is None:
        ground_truth = []

    fig = plt.figure(constrained_layout=True, figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, hspace=0.8, wspace=0.4)

    # Plot one TS by dimension
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.set_title("(a) Dimensions of Dataset: " + ds_name + "")

    # Plot bars to highlight similar motiflets
    ax_bars = fig.add_subplot(gs[1, 0], sharex=ax_ts)
    ax_bars.set_title("(b) Position of Top Motif Sets by Dimension")

    offset = 0
    tick_offsets = []
    ii = -1
    y_labels = []
    for dim in range(data_raw.shape[0]):
        dim_motiflets = candidates[dim, elbow_points[dim]]
        dim_data_raw = zscore(data_raw[dim, :])
        offset -= (np.max(dim_data_raw) - np.min(dim_data_raw))
        tick_offsets.append(offset)

        #  Plot the raw data
        _ = sns.lineplot(x=data_index,
                         y=dim_data_raw + offset,
                         ax=ax_ts,
                         linewidth=1,
                         color=color_palette[-1])

        sns.despine()

        #  Plot the motiflet within the TS
        for i, motiflet in enumerate(dim_motiflets):
            if motiflet is not None:
                for aa, pos in enumerate(motiflet):
                    _ = sns.lineplot(x=data_index[pos: pos + motif_length],
                                     y=dim_data_raw[pos: pos + motif_length] + offset,
                                     ax=ax_ts,
                                     linewidth=2,
                                     color=color_palette[0])

        for i, motiflet in enumerate(dim_motiflets):
            if motiflet is not None:
                color = color_palette[i]
                for aa, pos in enumerate(motiflet):
                    ratio = 0.8
                    rect = Rectangle(
                        (data_index[pos], -ii),  # (x,y)
                        data_index[pos + motif_length - 1] - data_index[pos],
                        ratio,
                        facecolor=color_palette[i],
                        # color_palette[dim % len(color_palette)],
                        alpha=0.5
                    )
                    ax_bars.add_patch(rect)
                if dimension_labels is not None:
                    y_labels.append(str(dimension_labels[dim])
                                    + " - Motif " + str(i + 1))
                else:
                    y_labels.append("Dim " + str(dim + 1) + " Motif " + str(i + 1))
                ii -= 1

    if dimension_labels is not None:
        ax_ts.set_yticks(tick_offsets)
        ax_ts.set_yticklabels(dimension_labels, fontsize=12)

    ax_bars.set_yticks(np.arange(len(y_labels)) + 1.5, )
    ax_bars.set_yticklabels(y_labels, fontsize=12)
    ax_bars.set_ylim([abs(ii) + 1, 1])

    if ground_truth is not None and len(ground_truth) > 0:
        ax_ts.legend(loc="upper left")

    plt.tight_layout()
    gs.tight_layout(fig)
    plt.show()


def plot_grid_motiflets(
        ds_name, data, candidates, elbow_points, dist,
        motif_length, font_size=20,
        ground_truth=None,
        show_elbows=False,
        dimension_labels=None,
        color_palette=sns.color_palette("tab10"),
        grid_dim=None,
        plot_index=None):
    """Plots the characteristic motifs for each method along the time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series data
    candidates: 2d array-like
        The motifset candidates
    elbow_points: array-like
        The elbow points found. Only motif sets from the elbow points will be plotted.
    dist: array-like
        The distances (extents) of the motif set candidates
    motif_length: int
        The motif length found.
    font_size: int
        Font-size to use for plotting.
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show_elbows: bool
        Show an elbow plot
    color_palette:
        Color-palette to use
    dimension_labels:
        Labels for the dimensions
    grid_dim: int
        The dimensionality of the grid (number of columns)
    plot_index: int
        Plots only the passed methods in the given order

    """

    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_context("paper",
                    rc={"font.size": font_size,
                        "axes.titlesize": font_size - 8,
                        "axes.labelsize": font_size - 8,
                        "xtick.labelsize": font_size - 10,
                        "ytick.labelsize": font_size - 10, })

    label_cols = 2

    count_plots = 4 if len(candidates[elbow_points]) > 6 else 3
    if show_elbows:
        count_plots = count_plots + 1

    if ground_truth is None:
        ground_truth = []

    if grid_dim is None:
        if plot_index is not None:
            ll = len(plot_index)
        else:
            ll = len(elbow_points)
        grid_dim = int(max(2, np.ceil(ll / 2)))

    dims = int(np.ceil(len(elbow_points) / grid_dim)) + count_plots

    fig = plt.figure(constrained_layout=True,
                     figsize=(10, dims * 3))
    gs = fig.add_gridspec(dims, grid_dim)

    ax_ts = fig.add_subplot(gs[0:2, :])
    ax_ts.set_title("(a) Dataset: " + ds_name + "")

    data_index, data_raw = ml.pd_series_to_numpy(data)

    # turn into 2d array
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))

    offset = 0
    tick_offsets = []
    motiflets = candidates[elbow_points]
    for dim in range(data_raw.shape[0]):
        dim_data_raw = zscore(data_raw[dim, :])
        offset -= (np.max(dim_data_raw) - np.min(dim_data_raw))
        tick_offsets.append(offset)

        #  Plot the raw data
        _ = sns.lineplot(x=data_index,
                         y=dim_data_raw + offset,
                         ax=ax_ts,
                         linewidth=1,
                         color=color_palette[0])
        ax_ts.set_yticklabels([], fontsize=12)
        sns.despine()

        #  Plot the motiflet
        for i, motiflet in enumerate(motiflets):
            if motiflet is not None:
                for aa, pos in enumerate(motiflet):
                    _ = sns.lineplot(x=data_index[pos: pos + motif_length],
                                     y=dim_data_raw[pos: pos + motif_length] + offset,
                                     ax=ax_ts,
                                     linewidth=2,
                                     color=color_palette[1 + i])

    if len(candidates[elbow_points]) > 6:
        ax_bars = fig.add_subplot(gs[2:4, :], sharex=ax_ts)
        next_id = 4
    else:
        ax_bars = fig.add_subplot(gs[2, :], sharex=ax_ts)
        next_id = 3

    ax_bars.set_title("(b) Position of Top Motif Sets")

    if show_elbows:
        ax_elbow = fig.add_subplot(gs[next_id, :])
        ax_elbow.set_title("(c) Significant Elbow Points on " + ds_name)
        ax_elbow.plot(range(len(np.sqrt(dist))), dist, "b", label="Extent")
        lim1 = plt.ylim()[0]
        lim2 = plt.ylim()[1]
        for elbow in elbow_points:
            ax_elbow.vlines(
                elbow, lim1, lim2,
                label=str(elbow) + "-Motiflet"
            )
        ax_elbow.set(xlabel='Size (k)', ylabel='Extent')
        ax_elbow.xaxis.set_major_locator(MaxNLocator(integer=True))

    gs = fig.add_gridspec(dims, grid_dim)

    #### Hack to add a subplot title
    ax_title = fig.add_subplot(gs[count_plots, :])

    if show_elbows:
        ax_title.set_title('(d) Shape of Top Motif Sets by Method', pad=0)
    else:
        ax_title.set_title('(c) Shape of Top Motif Sets by Method', pad=30)

    # Turn off axis lines and ticks of the big subplot 
    ax_title.tick_params(labelcolor=(1., 1., 1., 0.0),
                         top='off', bottom='off', left='off', right='off')
    ax_title.axis('off')
    ax_title._frameon = False
    sns.despine()

    y_labels = []
    ii = -1
    motiflets = candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        if motiflet is not None:

            plot_minature = (plot_index == None) or (i in plot_index)
            if plot_minature:
                ii = ii + 1
                off = int(ii / grid_dim)
                ax_motiflet = fig.add_subplot(gs[count_plots + off, ii % grid_dim])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for dim in range(data_raw.shape[0]):
                # for aa, pos in enumerate(motiflet):
                pos = motiflet[0]  # only take one subsequence
                normed_data = zscore(data_raw[dim, pos:pos + motif_length])
                df["_dim_" + str(dim)] = normed_data

            for aa, pos in enumerate(motiflet):
                ratio = 0.8
                rect = Rectangle(
                    (data_index[pos], -i),
                    data_index[pos + motif_length - 1] - data_index[pos],
                    ratio,
                    facecolor=color_palette[
                        (len(ground_truth) + ii % grid_dim) % len(color_palette)],
                    alpha=0.7
                )
                ax_bars.add_patch(rect)

            if plot_minature:
                df_melt = pd.melt(df, id_vars=["time"])
                _ = sns.lineplot(ax=ax_motiflet,
                                 data=df_melt,
                                 x="time", y="value", style="variable",
                                 # hue="variable",
                                 # ci=99, n_boot=10,
                                 ci=None, estimator=None,
                                 color=color_palette[0],
                                 )
                ax_motiflet.set_ylabel("")

                if isinstance(data, pd.Series):
                    ax_motiflet.set_xlabel(data.index.name)

                sns.despine()
                ax_motiflet.legend().set_visible(False)

            if plot_minature:
                ax_motiflet.set_yticks([])

    if dimension_labels is not None:
        ax_ts.set_yticks(tick_offsets)
        ax_ts.set_yticklabels(dimension_labels, fontsize=12)

    ax_bars.set_yticks(-np.arange(len(y_labels)) + 0.5, )
    ax_bars.set_yticklabels(y_labels, fontsize=12)
    ax_bars.set_ylim([-len(motiflets) + 1, 1])

    if ground_truth is not None and len(ground_truth) > 0:
        ax_ts.legend(loc="upper left", ncol=label_cols)

    plt.tight_layout()
    gs.tight_layout(fig)

    # plt.savefig(
    #    "video/motiflet_" + ds_name + "_Channels_" + str(len(df.index)) + "_Grid.pdf")

    plt.show()


def plot_all_competitors(
        data,
        ds_name,
        motifsets,
        motif_length,
        method_names=None,
        ground_truth=None,
        dimension_labels=None,
        plot_index=None,
        color_palette=sns.color_palette("tab10"),
        slack=0.5):
    """Plots the found motif sets of multiple competitor methods

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series data
    motifsets: 2d array-like
        The found motif sets for plotting
    motif_length: int
        The motif length found.
    method_names: array-like
        Names of the method to plot
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    dimension_labels:
        Labels for the dimensions
    grid_dim: int
        The dimensionality of the grid (number of columns)
    plot_index: int
        Plots only the passed methods in the given order
    """

    # convert to numpy array
    _, data_raw = ml.pd_series_to_numpy(data)
    D_full = ml.compute_distances_full(data_raw, motif_length, slack=slack)
    indices = np.arange(len(motifsets))

    dists = [ml.get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf)
             for motiflet_pos in motifsets]

    plot_grid_motiflets(
        ds_name, data, motifsets, indices,
        dists, motif_length,
        font_size=26,
        method_names=method_names,
        ground_truth=ground_truth,
        dimension_labels=dimension_labels,
        color_palette=color_palette,
        plot_index=plot_index)


def plot_competitors(
        data,
        ds_name,
        motifsets,
        motif_length,
        prefix="",
        filter=True,
        ground_truth=None,
        dimension_labels=None,
        slack=0.5):
    """Plots motif sets of a single competitor method.

    Parameters
    ----------
    data: array-like
        The time series data
    ds_name: String
        The name of the time series
    motifsets: array-like
        The motifset for plotting
    motif_length: int
        The motif length found.
    prefix: String
        The method name
    filter: bool, default=True
        filter overlapping motifs
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    dimension_labels:
        Labels for the dimensions

    """

    # convert to numpy array
    _, data_raw = ml.pd_series_to_numpy(data)

    D_full = ml.compute_distances_full(data_raw, motif_length, slack=slack)

    last = -1
    motifsets_filtered = []
    for motifset in motifsets:
        if ((len(motifset) > last) or (not filter)):
            motifsets_filtered.append(motifset)
            last = len(motifset)
    motifsets_filtered = np.array(motifsets_filtered)

    elbow_points = np.arange(len(motifsets_filtered))

    if filter:
        elbow_points = ml._filter_unique(elbow_points, motifsets_filtered, motif_length)

    dists = [ml.get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf)
             for motiflet_pos in motifsets_filtered]

    plot_grid_motiflets(
        ds_name, data, motifsets_filtered, elbow_points,
        dists, motif_length, method_name=prefix,
        dimension_labels=dimension_labels,
        ground_truth=ground_truth)

    return motifsets_filtered[elbow_points]


def format_key(e):
    key = ""
    if e > 0:
        key = "+" + str(e * 100) + "%"
    elif e < 0:
        key = str(e * 100) + "%"
    return key


def to_df(motif_sets, method_name, df, df2=None):
    df_all_1 = pd.DataFrame()
    df_all_2 = pd.DataFrame()
    for key in motif_sets:
        ms_set_finder = motif_sets[key]
        df_all_1[method_name + " Top-1 " + key] = [ms_set_finder[-1]]
        df[method_name + " Top-1 " + key] = [ms_set_finder[-1]]

        if df2 is not None:
            df_all_2[method_name + " Top-2 " + key] = [ms_set_finder[-2]]
            df2[method_name + " Top-2 " + key] = [ms_set_finder[-2]]

    if df2 is not None:
        df_all = (pd.concat([df_all_1, df_all_2], axis=1)).T
    else:
        df_all = df_all_1.T

    df_all.rename(columns={0: "offsets"}, inplace=True)
    return df_all
