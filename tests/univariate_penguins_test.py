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

def test_univariate_2():
    length = 2000
    ds_name, B = read_penguin_data()

    series = B.iloc[497699:497699 + length, np.array([0])].T
    ml = Motiflets(ds_name,
                   series,
                   elbow_deviation=1.1,
                   # slack = 0.6
                   )

    k_max = 50
    motif_length_range = np.arange(10, 60)

    _, all_minima = ml.fit_motif_length(
        k_max,
        motif_length_range,
        subsample=1,
        plot=False
    )
