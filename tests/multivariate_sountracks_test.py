import os
import matplotlib as mpl
from leitmotifs.lama import read_audio_from_dataframe

from audio.lyrics import *
from leitmotifs.competitors import *

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"
write_audio = False

datasets = {
    "Lord of the Rings Symphony - The Shire": {
        "ks": [4],
        "n_dims": 5,
        "motif_length": 301,
        "length_range_in_seconds": np.arange(5.75, 8.0, 0.25),
        "slack": 1.0,
        "audio_file_url": path_to_wav + "Lord of the Rings Symphony - The Shire.mp3",
        "pandas_file_url": path + "Lord-of-the-Rings-Symphony-The-Shire.csv",
    },
    "Star Wars - The Imperial March": {
        "ks": [5],
        "n_dims": 3,
        "motif_length": 357,
        "length_range_in_seconds": np.arange(3.0, 7.0, 0.25),
        "slack": 1.0,
        "audio_file_url": path_to_wav + "Star_Wars_The_Imperial_March_Theme_Song.mp3",
        "pandas_file_url": path + "Star_Wars_The_Imperial_March_Theme_Song.csv",
    },
    # "Harry Potter - Hedwigs Theme": {
    #     "ks": [10],
    #     "n_dims": 3,
    #     "motif_length": 357,
    #     "length_range_in_seconds": np.arange(3.0, 7.0, 0.25),
    #     "slack": 1.0,
    #     "audio_file_url": path_to_wav + "Harry Potter - Hedwigs Theme.mp3",
    #     "pandas_file_url": path + "Harry-Potter-Hedwigs-Theme.csv",
    # },
    # "Hans Zimmer - He is a Pirate": {
    #     "ks": [10],
    #     "n_dims": 8,
    #     "motif_length": 129,
    #     "slack": 1.0,
    #     "length_range_in_seconds": np.arange(3.0, 5, 0.25),
    #     "audio_file_url": path_to_wav + "Hans Zimmer - He is a Pirate.mp3",
    #     "pandas_file_url": path + "Hans-Zimmer-He-is-a-Pirate.csv",
    # },
    # "Hans Zimmer - Zoosters Breakout": {
    #     "ks": [10],
    #     "n_dims": 5,
    #     "motif_length": 129,
    #     "length_range_in_seconds": np.arange(3.0, 6.0, 0.25),
    #     "slack": 1.0,
    #     "audio_file_url": path_to_wav + "Hans Zimmer - Zoosters Breakout.mp3",
    #     "pandas_file_url": path + "Hans-Zimmer-Zoosters-Breakout.csv",
    # },
}


# dataset = datasets["Lord of the Rings Symphony - The Shire"]
# dataset = datasets["Star Wars - The Imperial March"]
# dataset = datasets["Hans Zimmer - Zoosters Breakout"]
# dataset = datasets["Hans Zimmer - He is a Pirate"]
# dataset = datasets["Harry Potter - Hedwigs Theme"]


def get_ds_parameters(name):
    global ks, n_dims, ds_name, slack, audio_file_url, motif_length
    global pandas_file_url, k_max, motif_length_range_in_s

    ds_name = name
    dataset = datasets[name]
    ks = dataset["ks"]
    n_dims = dataset["n_dims"]
    slack = dataset["slack"]
    audio_file_url = dataset["audio_file_url"]
    pandas_file_url = dataset["pandas_file_url"]
    motif_length = dataset["motif_length"]

    # for learning parameters
    k_max = np.max(dataset["ks"]) + 2
    motif_length_range_in_s = dataset["length_range_in_seconds"]


channels = [
    'MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
    'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
    'MFCC 11', 'MFCC 12', 'MFCC 13'
]


# def test_read_write():
#     audio_length_seconds, df, index_range = read_mp3(audio_file_url)
#     df.to_csv(pandas_file_url, compression='gzip')
# #   # read_audio_from_dataframe(pandas_file_url)


# def test_ground_truth():
#     get_ds_parameters("Lord of the Rings Symphony - The Shire")
#     audio_length_seconds, df, index_range, ground_truth \
#         = read_audio_from_dataframe(pandas_file_url, channels)
#
#     # motif_length_range = np.int32(motif_length_range_in_s /
#     #                              audio_length_seconds * df.shape[1])
#
#     # print("Positions:", index_range[ground_truth.loc[0][0]])
#
#     pos = [19.4, 28.5, 116.8, 126.0, 144.6]
#     length = 8.3
#
#     positions = []
#     for p in pos:
#         positions.append([index_range.searchsorted(p),
#                           index_range.searchsorted(p + length)])
#     print(positions)
#     positions = np.array([positions])
#
#     if os.path.isfile(audio_file_url):
#         # extract leitmotifs
#         for a, motif in enumerate(positions):
#             motif_length = motif[a, 1] - motif[a, 0]
#             print(motif_length)
#             length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
#
#             extract_audio_segment(
#                 df, ds_name, audio_file_url, "snippets",
#                 length_in_seconds, index_range, motif_length,
#                 np.array(motif)[:, 0], id=(a + 1))
#
#     ml = Motiflets(ds_name, df,
#                    dimension_labels=df.index,
#                    n_dims=n_dims,
#                    ground_truth=ground_truth
#                    )
#     ml.plot_dataset()


def test_publication():
    test_lama()
    test_emd_pca()
    test_mstamp()
    test_kmotifs()


def test_lama(
        dataset_name="Lord of the Rings Symphony - The Shire",
        minimize_pairwise_dist=False,
        use_PCA=False,
        motifset_name="LAMA",
        distance="znormed_ed",
        plot=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        series = pca.fit_transform(df.T).T
        df_transform = pd.DataFrame(series, index=["MFCC 0"], columns=df.columns)
    else:
        df_transform = df

    ml = LAMA(ds_name, df_transform,
              dimension_labels=df.index,
              distance=distance,
              n_dims=n_dims,
              slack=slack,
              minimize_pairwise_dist=minimize_pairwise_dist,
              ground_truth=ground_truth
              )

    # motif_length_range = np.int32(motif_length_range_in_s /
    #                               audio_length_seconds * df.shape[1])
    # motif_length, _ = ml.fit_motif_length(
    #     k_max,
    #     motif_length_range,
    #     plot=True,
    #     plot_elbows=True,
    #     plot_motifsets=True,
    #     plot_best_only=True,
    # )

    dists, motif_sets, elbow_points = ml.fit_k_elbow(
        k_max=k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifsets=False,
    )
    if plot:
        ml.elbow_points = ks
        ml.plot_motifset(motifset_name=motifset_name)

    if use_PCA:
        dims = [np.argsort(pca.components_[:])[:, :n_dims][0] for _ in ks]
    else:
        dims = ml.leitmotifs_dims[ks]

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.leitmotifs[eb])
        print("Positions:", index_range[motiflet])
        print("Positions:", list(zip(motiflet, motiflet + motif_length)))

        if write_audio:
            if os.path.isfile(audio_file_url):
                # extract motif sets
                extract_audio_segment(
                    df, ds_name, audio_file_url, "snippets",
                    length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))

        print("\tdims\t:", repr(dims))

    return motif_sets[ks], dims


def test_emd_pca(dataset_name="Lord of the Rings Symphony - The Shire", plot=True):
    return test_lama(dataset_name, use_PCA=True, motifset_name="PCA", plot=plot)


def test_mstamp(dataset_name="Star Wars - The Imperial March",
                plot=True, use_mdl=False):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    motif, dims = run_mstamp(
        df, ds_name, motif_length=motif_length,
        ground_truth=ground_truth, plot=plot, use_mdl=use_mdl, use_dims=n_dims)

    if write_audio:
        if os.path.isfile(audio_file_url):
            length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length, motif)

    return motif, dims


def test_kmotifs(dataset_name="Lord of the Rings Symphony - The Shire",
                 first_dims=True, plot=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    motif_sets = []
    used_dims = []
    for target_k in ks:
        motif, dims = run_kmotifs(
            df,
            ds_name,
            motif_length=motif_length,
            slack=slack,
            r_ranges=np.arange(10, 1000, 10),
            use_dims=n_dims if first_dims else df.shape[0],  # first dims or all dims
            target_k=target_k,
            ground_truth=ground_truth,
            plot=plot
        )
        used_dims.append(np.arange(dims))
        motif_sets.append(motif)

        if write_audio:
            length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
            print(f"Length in seconds: {length_in_seconds}")
            if os.path.isfile(audio_file_url):
                extract_audio_segment(
                    df, ds_name, audio_file_url, "snippets",
                    length_in_seconds, index_range, motif_length, motif[-1])

    return motif_sets, used_dims


def test_plot_spectrogram():
    audio_file_urls = \
        [
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_0.wav",
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_1.wav",
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_2.wav",
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_3.wav"
        ]

    plot_spectrogram(audio_file_urls)
    plt.show()


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)
    ax[0].set_title("Spectrogram of found Leitmotif", size=16)

    for i, audio_file_url in enumerate(audio_file_urls):
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[:, 0], data[:, 1]
            ax[i].specgram(left, Fs=samplingFrequency, cmap='Grays',
                           scale='dB', vmin=-10)
            ax[i].set_ylabel("Freq.")
            ax[i].set_ylim([0, 5000])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig("images_paper/audio/lotr-spectrogram.pdf")


def test_publication():
    dataset_names = [
        "Star Wars - The Imperial March",
        "Lord of the Rings Symphony - The Shire"
    ]
    method_names = [
        "LAMA",
        "LAMA (naive)",
        "mSTAMP+MDL",
        "mSTAMP",
        "EMD*",
        "K-Motifs (TOP-f)",
        "K-Motifs (all)",
        "LAMA (cid)",
        "LAMA (ed)",
        "LAMA (cosine)"
    ]

    file_prefix = "results_soundtracks"
    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        run_tests(
            dataset_name,
            ks=ks,
            method_names=method_names,
            test_lama=test_lama,
            test_mstamp=test_mstamp,
            test_emd_pca=test_emd_pca,
            test_kmotifs=test_kmotifs,
            file_prefix=file_prefix,
            plot=False
        )


def test_plot_results():
    dataset_names = [
        "Star Wars - The Imperial March",
        "Lord of the Rings Symphony - The Shire"
    ]

    method_names = [
        "LAMA",
        "LAMA (naive)",
        "mSTAMP+MDL",
        "mSTAMP",
        "EMD*",
        "K-Motifs (TOP-f)",
        "K-Motifs (all)",
        "LAMA (cid)",
        "LAMA (ed)",
        "LAMA (cosine)"
    ]
    results = []
    all_plot_names = {
        "_new": [
            "mSTAMP+MDL",
            "mSTAMP",
            "EMD*",
            "K-Motifs (all)",
            "LAMA",
        ], "_distances": [
            "LAMA",
            "LAMA (cid)",
            "LAMA (ed)",
            "LAMA (cosine)"
        ]
    }

    file_prefix = "results_soundtracks"

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        audio_length_seconds, df, index_range, ground_truth \
            = read_audio_from_dataframe(pandas_file_url, channels)

        eval_tests(
            dataset_name,
            ds_name,
            df,
            method_names,
            motif_length,
            ground_truth,
            all_plot_names,
            file_prefix,
            results,
            plot=True
        )

    pd.DataFrame(
        data=np.array(results),
        columns=["Dataset", "Method", "Precision", "Recall"]).to_csv(
        "results/soundtracks_precision.csv")