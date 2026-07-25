# -*- coding: utf-8 -*-
"""Plotting utilities.
"""

__author__ = ["patrickzib"]

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


LAMA = ml.LAMA
convert_to_2d = ml.convert_to_2d
as_series = ml.as_series


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
