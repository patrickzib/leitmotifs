import pandas as pd
import scipy.io as sio

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

path = "../datasets/experiments/"


def read_penguin_data_short():
    test = sio.loadmat(path + 'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T
    ds_name = "Penguins (Snippet)"
    return ds_name, series


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


def test_synthesize():
    ds_name, series = read_penguin_data()
    length = 1000
    series = series.iloc[497699:497699 + length].T

    data = series.values

    new_data = np.zeros((4, 2 * data.shape[-1]), dtype=np.float32)
    new_data[0, 0:data.shape[-1]] = data[0]
    new_data[1, 0:data.shape[-1]] = data[0]
    new_data[2, data.shape[-1]:] = data[2]
    new_data[3, data.shape[-1]:] = data[2]

    series = pd.DataFrame(new_data)

    ml = Motiflets(ds_name, series, n_dims=2)
    ml.plot_dataset()

    k_max = 40
    motif_length_range = np.arange(20, 35, 1)
    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=True,
        plot_motifsets=True,
        plot_best_only=True
    )

    #ml.plot_motifset()

    # dists[a] = ml.dists[ml.elbow_points[-1]]
    print("Best found length", best_length)
