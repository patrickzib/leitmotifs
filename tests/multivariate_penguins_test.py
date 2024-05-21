import scipy.io as sio

from leitmotifs.competitors import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"

# Ground Truth Parameters
lengths = [23, 21]  # As used by k-Motiflets
targets = [35, 24]  # As used by k-Motiflets
channels = [0, 1, 2, 3, 4, 5, 7]
ts_length = 1_000


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins"

    return ds_name, series


def read_penguin_data_short():
    test = sio.loadmat(path + 'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T
    ds_name = "Penguins (Snippet)"
    return ds_name, series


def test_plot_data():
    ds_name, series = read_penguin_data()
    series = series.iloc[497699 - 5000: 497699 + 5000, np.array([0, 7])].T

    ml = LAMA(ds_name, series)
    ml.plot_dataset()


def test_univariate():
    ds_name, series = read_penguin_data_short()

    ml = LAMA(
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
    ds_name, B = read_penguin_data()

    for start in [0, 2000]:
        series = B.iloc[
                 497699 + start:497699 + start + ts_length,
                 channels
                 ].T

        ml = LAMA(ds_name, series,
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
        ml.plot_motifset()

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
    ds_name = "Penguins"
    df = B.iloc[497699: 497699 + length, 0:7].T

    ml = LAMA(ds_name, df, elbow_deviation=1.25, slack=0.3)
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
    ml = LAMA(ds_name, series)

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
        series = B.iloc[497699 + start:497699 + start + length].T

        ml = LAMA(ds_name, series)
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


def test_lama(use_PCA=False):
    ds_name, B = read_penguin_data()
    n_dims = 2

    for start in [0, 3000]:
        series = B.iloc[497699 + start:497699 + start + ts_length, channels].T

        # make the signal uni-variate by applying PCA
        if use_PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            series = pca.fit_transform(series.T).T

        ml = LAMA(ds_name, series,
                  n_dims=n_dims,
                  n_jobs=8,
                  elbow_deviation=1.25,
                  )

        k_max = 40
        motif_length_range = np.arange(20, 30, 1)

        best_length, _ = ml.fit_motif_length(
            k_max,
            motif_length_range,
            plot=False,
            plot_elbows=True,
            plot_motifsets=False,
            plot_best_only=False
        )
        ml.plot_motifset(path="images_paper/penguins/penguins_" + str(start) + ".pdf")

        print("Positions:")
        for eb in ml.elbow_points:
            motiflet = np.sort(ml.leitmotifs[eb])
            print("\tpos\t:", repr(motiflet))

            if use_PCA:
                print("\tdims\t:", repr(np.argsort(pca.components_[:])[:, :n_dims]))
            else:
                print("\tdims\t:", repr(ml.leitmotif_dims[eb]))


def test_emd_pca():
    test_lama(use_PCA=True)


def test_mstamp():
    ds_name, B = read_penguin_data()
    for i, start in enumerate([0, 3000]):
        series = B.iloc[497699 + start:497699 + start + ts_length, channels].T

        for use_mdl in [True, False]:
            run_mstamp(series, ds_name, motif_length=lengths[i],
                          plot=True,
                          use_mdl=True, use_dims=2)



def test_kmotifs():
    ds_name, B = read_penguin_data()
    for i, start in enumerate([0, 3000]):
        series = B.iloc[497699 + start:497699 + start + ts_length, channels].T

        for first_dims in [True, False]:
            motif, dims = run_kmotifs(
                series,
                ds_name,
                motif_length=lengths[i],
                r_ranges=np.arange(0.1, 100, 0.5),
                use_dims=2 if first_dims else series.shape[0],  # first dims or all dims
                target_k=targets[i],
                plot=True
            )

            print("Name", "TOP-f" if first_dims else "all dims")
            print("Motif:\t", repr(motif))
            print("Dims: \t", repr(dims))



def test_plot_all():
    lengths = [23, 21]

    motif_sets = [
        [  # mstamp
            [346, 366],
            # leitmotifs
            [190, 209, 228, 247, 267, 287, 306, 326, 346, 366, 386, 406, 426, 446, 466,
             486, 506, 527, 547, 567, 587, 607, 628, 648, 669, 689, 710, 730, 768, 788,
             809, 851, 871, 936, 957],
            #
            # EMD*
            [91, 109]

        ],
        [  # mstamp
            [346, 366],
            # leitmotifs
            [22, 56, 92, 125, 158, 191, 227, 260, 291, 323, 357, 385, 418, 452, 479,
             511, 542, 573, 599, 620, 662, 706, 758, 792],
            # EMD*
            [21, 92]
        ]
    ]

    dims = [
        [  # mstamp
            [6],
            # leitmotifs
            [2, 0],
            # PCA + leitmotifs
            [0, 5]
        ],
        [  # mstamp
            [6],
            # leitmotifs
            [2, 0],
            # PCA + leitmotifs
            [1, 5]
        ]
    ]

    ds_name, B = read_penguin_data()
    for i, start in enumerate([0, 3000]):
        series = B.iloc[497699 + start:497699 + start + ts_length, channels]
        path = "images_paper/penguins/penguins_" + str(start) + "_new.pdf"

        motifset_names = ["mStamp + MDL", "Motiflets", "PCA+Univariate"]
        motifs = motif_sets[i]
        dim = dims[i]
        fig, ax = plot_motifsets(
            ds_name,
            series.T,
            motifsets=motifs,
            leitmotif_dims=dim,
            motifset_names=motifset_names,
            motif_length=lengths[i],
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()
