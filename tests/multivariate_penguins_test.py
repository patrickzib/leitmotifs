import scipy.io as sio

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

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


def test_univariate_top2():
    length = 4000
    ds_name, B = read_penguin_data()

    series = B.iloc[497699:497699 + length, np.array([0])].T
    series.reset_index(drop=True, inplace=True)

    ml = Motiflets(ds_name,
                   series,
                   elbow_deviation=1,
                   slack=0.8
                   )

    k_max = 50
    motif_length_range = np.arange(20, 45)

    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=False,
        plot_motifsets=False
    )
    ml.plot_motifset()
    print("Best found length", best_length)

    exclusion = ml.motiflets[ml.elbow_points[-1]]
    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        exclusion=exclusion,  # TODO: refactor?
        exclusion_length=best_length,
        plot=True,
        plot_elbows=False,
        plot_motifsets=False
    )
    ml.plot_motifset()


def test_multivariate():
    length = 2000
    ds_name, B = read_penguin_data()

    for start in [0]:  # , 2000
        # dists = np.zeros(5)
        series = B.iloc[497699 + start:497699 + start + length].T

        # for a, n_dims in enumerate(range(1, 6)):
        ml = Motiflets(ds_name, series,
                       n_dims=3
                       )

        k_max = 40
        motif_length_range = np.arange(20, 35, 1)

        best_length, _ = ml.fit_motif_length(
            k_max,
            motif_length_range,
            plot=True,
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
    sns.lineplot(x=np.arange(1, 6, dtype=np.int32), y=dists, ax = ax)
    plt.tight_layout()
    plt.show()


def test_multivariate_top2():
    length = 2000
    ds_name, B = read_penguin_data()

    series = B.iloc[497699:497699 + length, np.array([0, 2])].T
    ml = Motiflets(ds_name, series,
                   slack=0.8, n_dims=2
                   )

    k_max = 50
    motif_length_range = np.arange(15, 35, 1)

    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot_motifsets=False,
    )
    ml.plot_motifset()

    print("Best found length", best_length)

    exclusion = ml.motiflets[ml.elbow_points]

    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot_elbows=True,
        plot_motifsets=False,
        exclusion=exclusion,
        exclusion_length=best_length,
    )
    ml.plot_motifset()

    print("Best found length", best_length)

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

    for start in [0]: # , 2000
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
    n = 30_000
    series = series.iloc[497699:497699 + n:, 0:2].T.to_numpy()

    m = 1000
    k = 10
    D_knn, D_sparse, knns = ml.compute_distance_matrix_sparse(series, m=m, k=k)

    elements = 0
    for A in D_sparse:
        for B in A:
            elements += len(B)

    n = (series.shape[1]-m+1)
    print(elements, series.shape[0] * (n**2),
          str(elements * 100 / (series.shape[0] * (n**2))) + "%")


def test_full():
    ds_name, series = read_penguin_data()
    series = series.iloc[497699:497699 + 10000].T

    m = 100
    _, _ = ml.compute_distance_matrix(series.to_numpy(), m=m, k=5)
