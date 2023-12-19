import scipy.io as sio

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


def read_penguin_data_short():
    test = sio.loadmat(path + 'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T
    ds_name = "Penguins (Snippet)"
    return ds_name, series


def test_plot_data():
    ds_name, series = read_penguin_data()
    series = series.iloc[497699 - 5000: 497699 + 5000, np.array([0, 7])].T

    ml = Motiflets(ds_name, series)
    ml.plot_dataset()


def test_univariate():
    ds_name, series = read_penguin_data_short()

    ml = Motiflets(
        ds_name, series,
        elbow_deviation=1,
        slack=0.8
    )
    # ml.plot_dataset()

    k_max = 50
    motif_length_range = np.arange(15, 50, 1)

    _, all_minima = ml.fit_motif_length(
        k_max, motif_length_range,
        plot_elbows=False,
        plot_motifsets=False,
        plot_best_only=True,
        subsample=1)

    ml.plot_motifset()


def test_multivariate():
    length = 1_000
    ds_name, B = read_penguin_data()

    for start in [0, 2000]:
        # dists = np.zeros(5)
        series = B.iloc[497699 + start:497699 + start + length, [0, 1, 2, 3, 4, 5, 7]].T

        ml = Motiflets(ds_name, series,
                       n_dims=2,
                       n_jobs=8,
                       )

        k_max = 40
        motif_length_range = np.arange(20, 30, 1)

        best_length, _ = ml.fit_motif_length(
            k_max,
            motif_length_range,
            plot=False,
            plot_elbows=False,
            plot_motifsets=True,
            plot_best_only=True
        )
        # ml.plot_motifset()

        # dists[a] = ml.dists[ml.elbow_points[-1]]
        print("Best found length", best_length)

        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.set_title("Dimension Plot")
        # sns.lineplot(x=np.arange(1, 6, dtype=np.int32), y=dists, ax = ax)
        # plt.tight_layout()
        # plt.show()


def test_plot_n_dim_plot():
    dists = [0., 25.16639805, 89.46489525, 137.10974884, 195.87618828]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Dimension Plot")
    sns.lineplot(x=np.arange(1, 6, dtype=np.int32), y=dists, ax=ax)
    plt.tight_layout()
    plt.show()


def test_univariate_profile():
    # ds_name, series = read_penguin_data_short()
    length = 5000
    B = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    df = B.iloc[497699: 497699 + length, 0:7].T

    ml = Motiflets(ds_name, df, elbow_deviation=1.25, slack=0.3)
    # ml.plot_dataset()

    k_max = 50
    motif_length_range = np.arange(10, 30, 1)

    _, all_minima = ml.fit_motif_length(
        k_max, motif_length_range,
        plot=False, plot_elbows=False, plot_motifsets=False,
        subsample=1)

    # ml.plot_motifset()


def test_multivariate_all():
    length = 2000
    ds_name, B = read_penguin_data()

    series = B.iloc[497699:497699 + length, 0:3].T
    ml = Motiflets(ds_name, series)

    k_max = 30
    motif_length_range = np.arange(20, 35, 1)

    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=False,
        plot_elbows=True,
        plot_motifsets=False,
    )
    ml.plot_motifset()

    print("Best found length", best_length)


def test_fit_dimensions():
    length = 2000
    ds_name, B = read_penguin_data()

    for start in [0]:  # , 2000
        dists = np.zeros(5)
        series = B.iloc[497699 + start:497699 + start + length].T

        ml = Motiflets(ds_name, series)
        k_max = 40
        ml.fit_dimensions(
            k_max,
            motif_length=22,
            dim_range=np.arange(1, 6, dtype=np.int32),
        )


def test_sparse():
    ds_name, series = read_penguin_data()
    n = 100_000  # 100k equals 2 mins, 200k equals 13 mins
    series = series.iloc[497699:497699 + n:, 0:3].T.to_numpy()

    m = 100
    k = 10
    D_knn, D_sparse, knns = ml.compute_distance_matrix_sparse(series, m=m, k=k)

    elements = 0
    for A in D_sparse:
        for B in A:
            elements += len(B)

    n = (series.shape[1] - m + 1)
    print(elements, series.shape[0] * (n ** 2),
          str(elements * 100 / (series.shape[0] * (n ** 2))) + "%")


def test_full():
    ds_name, series = read_penguin_data()
    n = 30_000
    series = series.iloc[497699:497699 + n:, 0:3].T.to_numpy()

    m = 100
    k = 10

    _, _ = ml.compute_distance_matrix(series, m=m, k=k)


def test_publication():
    length = 1_000
    ds_name, B = read_penguin_data()

    for start in [0, 3000]:
        series = B.iloc[497699 + start:497699 + start + length, [0, 1, 2, 3, 4, 5, 7]].T

        ml = Motiflets(ds_name, series,
                       n_dims=2,
                       n_jobs=8,
                       elbow_deviation=1.25,
                       )

        k_max = 40
        motif_length_range = np.arange(20, 30, 1)

        best_length, _ = ml.fit_motif_length(
            k_max,
            motif_length_range,
            plot=True,
            plot_elbows=True,
            plot_motifsets=False,
            plot_best_only=False
        )
        ml.plot_motifset(path="images_paper/penguins/penguins_" + str(start) + ".pdf")

        print("Positions:")
        for eb in ml.elbow_points:
            motiflet = np.sort(ml.motiflets[eb])
            print("\tpos\t:", motiflet)
            print("\tdims\t:", ml.motiflets_dims[eb])


def test_mstamp():
    import stumpy

    length = 1_000
    ds_name, B = read_penguin_data()
    lengths = [23, 21]
    for i, start in enumerate([0, 3000]):
        series = B.iloc[497699 + start:497699 + start + length, [0, 1, 2, 3, 4, 5, 7]]

        m = lengths[i]  # As used by k-Motiflets

        # Find the Pair Motif
        mps, indices = stumpy.mstump(series, m=m)
        motifs_idx = np.argmin(mps, axis=1)
        nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

        # Find the optimal dimensionality by minimizing the MDL
        mdls, subspaces = stumpy.mdl(series, m, motifs_idx, nn_idx)

        plt.plot(np.arange(len(mdls)), mdls, c='red', linewidth='2')
        plt.xlabel('k (zero-based)')
        plt.ylabel('Bit Size')
        plt.xticks(range(mps.shape[0]))
        plt.tight_layout()
        plt.show()

        k = np.argmin(mdls)
        print("Best dimensions", series.columns[subspaces[k]])

        # found Pair Motif
        motif = [motifs_idx[subspaces[k]][0], nn_idx[subspaces[k]][0]]
        print("Pair Motif Position:")
        print("\tpos:\t", motif)
        print("\tf:  \t", subspaces[k])

        dims = [subspaces[k]]
        motifs = [[motifs_idx[subspaces[k]][0], nn_idx[subspaces[k]][0]]]
        motifset_names = ["mStamp"]

        fig, ax = plot_motifsets(
            ds_name,
            series.T,
            motifsets=motifs,
            motiflet_dims=dims,
            motifset_names=motifset_names,
            motif_length=m,
            show=True)


def test_plot_both():
    lengths = [23, 21]

    motif_sets = [
        [  # mstamp
            [346, 366],
            # motiflets
            [190, 209, 228, 247, 267, 287, 306, 326, 346, 366, 386, 406, 426, 446, 466,
             486, 506, 527, 547, 567, 587, 607, 628, 648, 669, 689, 710, 730, 768, 788,
             809, 851, 871, 936, 957]
        ],
        [  # mstamp
            [346, 366],
            # motiflets
            [22, 56, 92, 125, 158, 191, 227, 260, 291, 323, 357, 385, 418, 452, 479,
             511, 542, 573, 599, 620, 662, 706, 758, 792]
        ]
    ]

    dims = [
        [  # mstamp
            [6],
            # motiflets
            [2, 0]

        ],
        [  # mstamp
            [6],
            # motiflets
            [2, 0]
        ]
    ]

    length = 1_000
    ds_name, B = read_penguin_data()
    for i, start in enumerate([0, 3000]):
        series = B.iloc[497699 + start:497699 + start + length, [0, 1, 2, 3, 4, 5, 7]]

        m = lengths[i]  # As used by k-Motiflets

        path = "images_paper/penguins/penguins_" + str(start) + ".pdf"

        motifset_names = ["mStamp + MDL", "Motiflets"]
        motifs = motif_sets[i]
        dim = dims[i]
        fig, ax = plot_motifsets(
            ds_name,
            series.T,
            motifsets=motifs,
            motiflet_dims=dim,
            motifset_names=motifset_names,
            motif_length=lengths[i],
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()



def test_map():
    import scipy.stats
    length = 1_000
    ds_name, B = read_penguin_data()
    for i, start in enumerate([0, 3000]):
        series = B.iloc[497699 + start:497699 + start + length, [0, 1, 2, 3, 4, 5, 7]]


        # print(scipy.stats.median_absolute_deviation(series, axis=0))

        print("Normal\t", scipy.stats.median_abs_deviation(series, axis=0, scale="normal"))

        print("Constant\t", scipy.stats.median_abs_deviation(series, axis=0, scale=1/1.4826))

        print("Default\t", scipy.stats.median_abs_deviation(series, axis=0))

        print("...")