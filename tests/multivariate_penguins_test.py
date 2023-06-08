from motiflets.plotting import *

import scipy.io as sio

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings
warnings.simplefilter("ignore")

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/chains/"

def test_univariate():
    test = sio.loadmat(path+'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T

    ds_name = "Penguins (Snippet)"
    plot_dataset(ds_name, series)

    ks = 50
    motif_length = 22
    dists, motiflets, elbow_points = plot_elbow(
        ks, series,
        ds_name=ds_name, plot_elbows=True,
        motif_length=motif_length,
        slack=0.5, elbow_deviation=1.25
    )


def test_multivariate():
    length = 1000
    B = pd.read_csv(path+"penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    for start in [0, 2000]:
        series = B.iloc[497699+start:497699+start+length, [0,1,2]].T
        # plot_dataset(ds_name, series)

        ks = 60
        motif_length = int(22)
        dists, motiflets, elbow_points = plot_elbow(
            ks, series,
            ds_name=ds_name,
            plot_elbows=False,
            plot_grid=False,
            motif_length=motif_length,
            # slack=0.6,
            elbow_deviation=1.1
        )

        plot_motifset(
            ds_name,
            series,
            motifset=motiflets[elbow_points[-1]],
            dist=dists[elbow_points[-1]],
            motif_length=motif_length,
            show=True)

