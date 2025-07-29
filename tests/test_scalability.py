import warnings

from leitmotifs.plotting import *

warnings.simplefilter("ignore")

path = "../datasets/experiments/"


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


def test_penguins_univ():
    lengths = [1_000,
               5_000,
               10_000
               # , 30_000,
               ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length)
        series = B.iloc[:length, 0].T

        ml = LAMA(ds_name,
                  series,
                  n_jobs=-1,
                  )

        k_max = 5

        t_before = time.time()
        _ = ml.fit_k_elbow(
            k_max,
            22,
            plot_elbows=False,
        )
        t_after = time.time()
        time_s[i] = t_after - t_before
        print("Time:", time_s[i])


def test_penguins_multivariate():
    lengths = [1_000,
               5_000,
               10_000,
               # 30_000,
               # 50_000,
               # 100_000,
               # 150_000, 200_000,
               # 250_000
               ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length, flush=True)
        series = B.iloc[:length].T

        ml = LAMA(
            ds_name,
            series,
            n_dims=3,
            backend="scalable",
            n_jobs=-1,
        )

        k_max = 5

        t_before = time.time()
        _ = ml.fit_k_elbow(
            k_max,
            motif_length=22,
            plot_elbows=False,
            plot_motifsets=False
        )
        t_after = time.time()
        time_s[i] = t_after - t_before
        memory_usage = ml.memory_usage

        print("\tTime:", time_s[i])
        print("\tMemory:", memory_usage, "MB")
