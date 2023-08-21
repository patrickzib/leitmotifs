import pandas as pd

import motiflets.motiflets as mof
from motiflets.plotting import *


def test_ecg():
    file = 'ecg-heartbeat-av.csv'
    ds_name = "ECG Heartbeat"
    series, df_gt = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df,
                   # elbow_deviation=1.25,
                   slack=0.25,
                   ground_truth=df_gt
                   )

    ml.plot_dataset()

    k_max = 20
    length_range = np.arange(50, 150, 5)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=1)

    print(all_minima, length_range[all_minima])

    # for motif_length in length_range[all_minima]:
    #    ml.fit_k_elbow(k_max, motif_length,
    #                   plot_elbows=True,
    #                   plot_motifs_as_grid=True)
    #
    #    # ml.plot_motifset()



def test_muscle_activation():
    file = 'muscle_activation.csv'
    ds_name = "Muscle Activation"

    series, df_gt = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df,
                   ground_truth=df_gt,
                   # elbow_deviation=1.25,
                   slack=0.4
                   )

    ml.plot_dataset()

    k_max = 20
    length_range = np.arange(400, 700, 25)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    print(all_minima, length_range[all_minima])

    #for motif_length in length_range[all_minima]:
    #    ml.fit_k_elbow(k_max, motif_length,
    #                   plot_elbows=True,
    #                   plot_motifs_as_grid=True)
    #    ml.plot_motifset()


def test_physiodata():
    """ Funktioniert: 2 Motifs variabler Länge """
    file = 'npo141.csv'  # Dataset Length n:  269286
    ds_name = "EEG Sleep Data"
    series = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    k_max = 50
    length_range = np.arange(10, 100, 5)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()


def test_winding():
    file = "winding_col.csv"
    ds_name = "Industrial winding process"
    series = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    k_max = 12
    length_range = np.arange(50, 100, 1)

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()

def test_fnirs():
    file = "fNIRS_subLen_600.csv"
    ds_name = "fNIRS"
    series = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    k_max = 20
    length_range = np.arange(20, 200, 5)

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()


def test_insects():
    ds_name = "Insect"
    data = pd.read_csv("../datasets/experiments/insect.csv",
                       squeeze=True)
    print("Dataset Original Length n: ", len(data))
    data, factor = mof._resample(data[:100000], sampling_factor=25000)
    data[:] = zscore(data)
    df = pd.DataFrame(data).T

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    k_max = 10
    length_range = np.arange(25, 125, 5)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()
