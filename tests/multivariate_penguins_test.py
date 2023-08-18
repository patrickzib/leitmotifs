import matplotlib
import scipy.cluster.hierarchy as sch
import scipy.io as sio

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/chains/"


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
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

    ml = Motiflets(ds_name, series, elbow_deviation=1.25)
    ml.plot_dataset()


def test_univariate():
    ds_name, series = read_penguin_data_short()

    ml = Motiflets(ds_name, series, elbow_deviation=1.25)
    ml.plot_dataset()

    k_max = 50
    motif_length_range = np.arange(10, 35, 1)

    _, all_minima = ml.fit_motif_length(
        k_max, motif_length_range,
        subsample=1)

    # motif_length = 22
    for minimum in all_minima:
        motif_length = motif_length_range[minimum]
        ml.fit_k_elbow(
            k_max,
            motif_length=motif_length,
            plot_elbows=True,
            plot_motifs_as_grid=False
        )

        ml.plot_motifset()


def test_multivariate():
    length = 1000
    ds_name, B = read_penguin_data()

    for start in [0, 2000]:
        series = B.iloc[497699 + start:497699 + start + length,
                 np.array([0, 1, 2, 3])].T
        ml = Motiflets(ds_name, series,
                       elbow_deviation=1.1,
                       # slack = 0.6
                       )
        # ml.plot_dataset()

        k_max = 50
        motif_length_range = np.arange(10, 35, 1)

        _, _ = ml.fit_motif_length(
            k_max,
            motif_length_range,
            subsample=1
        )

        motif_length = 22
        ml.fit_k_elbow(
            k_max,
            plot_elbows=True,
            plot_motifs_as_grid=False,
            motif_length=motif_length,
        )

        # path = "penguin/" + ds_name + "_start_" + str(start) + ".pdf"
        ml.plot_motifset()  # path


def test_dendrogram():
    length = 1000
    B = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    df = B.iloc[497699: 497699 + length, 0:7].T

    k_max = 60
    motif_length = 22

    ml = Motiflets(ds_name, df,
                   elbow_deviation=1.25,
                   slack=1.0,
                   dimension_labels=df.index
                   )

    ml.fit_dendrogram(k_max, motif_length, n_clusters=2)
