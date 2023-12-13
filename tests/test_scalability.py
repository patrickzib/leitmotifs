import scipy.io as sio
import pandas as pd
import stumpy

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


def test_motiflets():
    lengths = [1_000, 5_000,
               # 10_000, 30_000,
               # 50_000, 100_000,
               # 150_000, 200_000,
               # 250_000
               ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length)
        series = B.iloc[:length].T

        ml = Motiflets(ds_name, series,
                       n_dims=2,
                       n_jobs=8,
                       )

        k_max = 5

        t_before = time.time()
        _ = ml.fit_k_elbow(
            k_max,
            22,
            plot_elbows=False,
            plot_motifs_as_grid=False
        )
        t_after = time.time()
        time_s[i] = t_after - t_before
        print("Time:", time_s[i])

        dict = time_s
        df = pd.DataFrame(data=dict, columns=['Time'], index=lengths)
        df["Method"] = "k-Motiflets"
        df.index.name = "Lengths"
        df.to_csv('csv/scalability_motiflets_k5.csv')


def test_mstamp():
    lengths = [1_000, 5_000,
               10_000, 30_000,
               50_000, 100_000,
               # 150_000, 200_000,
               # 250_000
               ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length)
        series = B.iloc[:length].T

        t_before = time.time()
        mps, indices = stumpy.mstump(series.values, m=22)
        motifs_idx = np.argmin(mps, axis=1)
        nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

        t_after = time.time()
        time_s[i] = t_after - t_before
        print("Time:", time_s[i])

        dict = time_s
        df = pd.DataFrame(data=dict, columns=['Time'], index=lengths)
        df["Method"] = "k-Motiflets"
        df.index.name = "Lengths"
        df.to_csv('csv/mstamp.csv')


def test_sizes():
    n = 20000
    d = 8
    sparse_gb = ((n ** 2) * d) * 32 / (1024 ** 3) / 8
    gb = 4  # 4 GB
    print(sparse_gb, gb)


def test_plot():
    df = pd.read_csv("csv/scalability_motiflets_k5.csv", index_col=0)
    df.index.name = "Walltime in s"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Scalability in length n for d=8")
    sns.lineplot(data=df, ax=ax)
    plt.tight_layout()

    plt.savefig("images_paper/scalability/scalability_motiflets_k5.pdf")
    plt.show()
