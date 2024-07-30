import stumpy

from leitmotifs.competitors import *
from leitmotifs.plotting import *
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"


def read_penguin_data():
    series = pd.read_csv(
        path + "penguin.txt",
        names=(["X-Acc", "Y-Acc", "Z-Acc",
                "4", "5", "6",
                "7", "Pressure", "9"]),
        delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


def test_emd():
    test_lama(True)


def test_lama(use_PCA=False):
    lengths = [
        1_000, 5_000,
        10_000, 30_000,
        50_000, 100_000,
        150_000, 200_000,
        250_000
    ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length)
        series = B.iloc[:length].T

        # make the signal uni-variate by applying PCA
        if use_PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            df_transform = pca.fit_transform(series.T).T
        else:
            df_transform = df

        ml = LAMA(ds_name,
                  df_transform,
                  n_dims=2,
                  n_jobs=8,
                  )

        k_max = 5

        t_before = time.time()
        _ = ml.fit_k_elbow(
            k_max,
            22,
            plot_elbows=False,
            plot_motifsets=False
        )
        t_after = time.time()
        time_s[i] = t_after - t_before
        print("Time:", time_s[i])

        dict = time_s
        df = pd.DataFrame(data=dict, columns=['Time'], index=lengths)
        df.index.name = "Lengths"
        if use_PCA:
            df["Method"] = "EMD*"
            df.to_csv('csv/scalability_emd_k5.csv')
        else:
            df["Method"] = "Leitmotif"
            df.to_csv('csv/scalability_lama_k5.csv')


def test_mstamp():
    lengths = [
        1_000, 5_000,
        # 10_000, 30_000,
        # 50_000, 100_000,
        # 150_000,
        # 200_000,
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
        df["Method"] = "mSTAMP"
        df.index.name = "Lengths"
        df.to_csv('csv/mstamp2.csv')


def test_kmotif(first_dims=True):
    lengths = [
        1_000, 5_000,
        10_000, 20_000,
        # 50_000, 100_000,  # Out of Memory
        # 150_000,
        # 200_000,
        # 250_000
    ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length)
        series = B.iloc[:length].T

        t_before = time.time()

        run_kmotifs(
            series,
            ds_name,
            motif_length=22,
            r_ranges=np.arange(1, 10, 1),
            use_dims=2 if first_dims else df.shape[0],
            target_k=5,
            plot=False
        )

        t_after = time.time()
        time_s[i] = t_after - t_before
        print("Time:", time_s[i])

        dict = time_s
        df = pd.DataFrame(data=dict, columns=['Time'], index=lengths)
        df["Method"] = "K-Motif"
        df.index.name = "Lengths"
        name = "k_motif" + ("_first_dims" if first_dims else "")
        df.to_csv(f'csv/{name}.csv')


def test_plot():
    import matplotlib as mpl
    mpl.rcParams['lines.markersize'] = 12

    df = pd.read_csv("csv/scalability2.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Scalability in length n for d=8")
    sns.lineplot(x="Lengths", y="Time", hue="Method", style="Method", markers=True,
                 data=df.reset_index(), ax=ax)

    ax.annotate('Out-of-Memory', xy=(20000,10), xytext=(10000,1000),
                     arrowprops=dict(facecolor='red', arrowstyle="->",
                                     connectionstyle="arc3,rad=.5"),
                     fontsize=12, fontfamily='monospace', color="red", ha='left');

    # plt.yscale('log',base=2)
    ax.set_ylabel("Walltime in s")
    ax.set_xlabel("Time Series Length")
    plt.tight_layout()

    plt.savefig("images_paper/scalability/scalability.pdf")
    plt.show()

# Write to CHIME directory
# def test_write_data():
#     lengths = [1_000, 5_000,
#                10_000, 30_000,
#                50_000, 100_000,
#                150_000, 200_000,
#                250_000
#                ]
#
#     ds_name, B = read_penguin_data()
#
#     for i, length in enumerate(lengths):
#         series = B.iloc[:length].T
#         series.T.to_csv(
#             '/Users/bzcschae/workspace/multidim_motifs/CHIME/jar/penguin_' + str(
#                 length) + '.csv',
#             header=None, index=None)
