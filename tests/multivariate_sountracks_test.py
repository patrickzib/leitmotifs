import os
import matplotlib as mpl
from motiflets.motiflets import read_audio_from_dataframe
from motiflets.motiflets import read_ground_truth

from audio.lyrics import *
from motiflets.competitors import *

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

datasets = {
    "Hans Zimmer - He is a Pirate": {
        "ks": [10],
        "n_dims": 8,
        "motif_length": 129,
        "slack": 1.0,
        "length_range_in_seconds": np.arange(3.0, 5, 0.25),
        "ds_name": "Hans Zimmer - He is a Pirate",
        "audio_file_url": path_to_wav + "Hans Zimmer - He is a Pirate.mp3",
        "pandas_file_url": path + "Hans-Zimmer-He-is-a-Pirate.csv",
    },
    "Hans Zimmer - Zoosters Breakout": {
        "ks": [10],
        "n_dims": 5,
        "motif_length": 129,
        "length_range_in_seconds": np.arange(3.0, 6.0, 0.25),
        "slack": 1.0,
        "ds_name": "Hans Zimmer - Zoosters Breakout",
        "audio_file_url": path_to_wav + "Hans Zimmer - Zoosters Breakout.mp3",
        "pandas_file_url": path + "Hans-Zimmer-Zoosters-Breakout.csv",
    },
    "Lord of the Rings Symphony - The Shire": {
        "ks": [4],
        "n_dims": 5,
        "motif_length": 301,
        "length_range_in_seconds": np.arange(7.0, 8.0, 0.25),
        "slack": 1.0,
        "ds_name": "Lord of the Rings Symphony - The Shire",
        "audio_file_url": path_to_wav + "Lord of the Rings Symphony - The Shire.mp3",
        "pandas_file_url": path + "Lord-of-the-Rings-Symphony-The-Shire.csv",
    },

    "Star Wars - The Imperial March": {
        "ks": [5],
        "n_dims": 3,
        "motif_length": 357,
        "length_range_in_seconds": np.arange(3.0, 7.0, 0.25),
        "slack": 1.0,
        "ds_name": "Star Wars - The Imperial March",
        "audio_file_url": path_to_wav + "Star_Wars_The_Imperial_March_Theme_Song.mp3",
        "pandas_file_url": path + "Star_Wars_The_Imperial_March_Theme_Song.csv",
    },
    "Harry Potter - Hedwigs Theme": {
        "ks": [10],
        "n_dims": 3,
        "motif_length": 357,
        "length_range_in_seconds": np.arange(3.0, 7.0, 0.25),
        "slack": 1.0,
        "ds_name": "Harry Potter - Hedwigs Theme",
        "audio_file_url": path_to_wav + "Harry Potter - Hedwigs Theme.mp3",
        "pandas_file_url": path + "Harry-Potter-Hedwigs-Theme.csv",
    },
}

dataset = datasets["Lord of the Rings Symphony - The Shire"]
# dataset = datasets["Star Wars - The Imperial March"]

# dataset = datasets["Hans Zimmer - Zoosters Breakout"]
# dataset = datasets["Hans Zimmer - He is a Pirate"]
# dataset = datasets["Harry Potter - Hedwigs Theme"]


ks = dataset["ks"]
n_dims = dataset["n_dims"]
ds_name = dataset["ds_name"]
slack = dataset["slack"]
audio_file_url = dataset["audio_file_url"]
pandas_file_url = dataset["pandas_file_url"]
m = dataset["motif_length"]

# for learning parameters
k_max = np.max(dataset["ks"]) + 2
motif_length_range_in_s = dataset["length_range_in_seconds"]

channels = [
    'MFCC 0',
    'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
    # 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
    # 'MFCC 11', 'MFCC 12', 'MFCC 13'
]


# def test_read_write():
#     audio_length_seconds, df, index_range = read_mp3(audio_file_url)
#     df.to_csv(pandas_file_url, compression='gzip')
# #   # read_audio_from_dataframe(pandas_file_url)


def test_ground_truth():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    # motif_length_range = np.int32(motif_length_range_in_s /
    #                              audio_length_seconds * df.shape[1])

    # print("Positions:", index_range[ground_truth.loc[0][0]])

    pos = [19.4, 28.5, 116.8, 126.0, 144.6]
    length = 8.3

    positions = []
    for p in pos:
        positions.append([index_range.searchsorted(p),
                          index_range.searchsorted(p+length)])
    print(positions)
    positions = np.array([positions])

    if os.path.isfile(audio_file_url):
        # extract motiflets
        for a, motif in enumerate(positions):
            motif_length = motif[a, 1] - motif[a, 0]
            print(motif_length)
            length_in_seconds = motif_length * audio_length_seconds / df.shape[1]

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length,
                np.array(motif)[:, 0], id=(a + 1))

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth
                   )
    ml.plot_dataset()

def test_publication():
    test_lama()
    test_emd_pca()
    test_mstamp()
    test_kmotifs()


def test_lama(use_PCA=False):
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        series = pca.fit_transform(df.T).T
        df = pd.DataFrame(series, index=["MFCC 0"], columns=df.columns)

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   slack=1.0,
                   ground_truth=ground_truth
                   )

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])
    # m, _ = ml.fit_motif_length(
    #     k_max,
    #     motif_length_range,
    #     plot=True,
    #     plot_elbows=False,
    #     plot_motifsets=True,
    #     plot_best_only=False,
    # )

    m = 290
    ks = [6]
    ml.fit_k_elbow(
       k_max=k_max,
       motif_length=m,
       plot_elbows=False,
       plot_motifsets=False,
    )
    ml.elbow_points = ks
    ml.plot_motifset(
       path="images_paper/audio/" + ds_name + "_new.pdf",
       motifset_name="LAMA")

    # m = motif_length
    length_in_seconds = m * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, m)

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:", index_range[motiflet])
        print("Positions:", list(zip(motiflet, motiflet + m)))

        #if False:
        if os.path.isfile(audio_file_url):
            # extract motiflets
            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, m, motiflet, id=(a + 1))

        if use_PCA:
            print("\tdims\t:", repr(np.argsort(pca.components_[:])[:, :n_dims]))
        else:
            print("\tdims\t:", repr(ml.motiflets_dims[eb]))


def test_emd_pca():
    test_lama(use_PCA=True)


def test_mstamp():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    motif = run_mstamp(df, ds_name, motif_length=m, ground_truth=ground_truth)

    if os.path.isfile(audio_file_url):
        length_in_seconds = m * audio_length_seconds / df.shape[1]
        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets",
            length_in_seconds, index_range, m, motif)


def test_kmotifs():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    for target_k in ks:
        motif = run_kmotifs(
            df,
            ds_name,
            m,
            r_ranges=np.arange(10, 1000, 10),
            use_dims=n_dims, # df.shape[0],
            target_k=target_k,
            slack=slack,
            ground_truth=ground_truth
        )

        length_in_seconds = m * audio_length_seconds / df.shape[1]
        print(f"Length in seconds: {length_in_seconds}")

        if os.path.isfile(audio_file_url):
            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, m, motif[-1])


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


def test_plot_all():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    path = "images_paper/audio/" + ds_name + "_new.pdf"

    motifs = [
        # mstamp
        [2666, 2860],
        # LAMA
        [287, 664, 5409, 5866],
        # EMD*
        [715, 2047, 2460, 4249, 5468, 5926],
        # K-Motif,
        [706, 4212, 5460, 5923]
    ]

    dims = [
        # mstamp
        [0],
        # LAMA
        [13, 7, 1, 10, 6],
        # EMD*
        [1, 0, 3, 9, 11],
        # K-Motif
        np.arange(5)
    ]

    motifset_names = ["mStamp + MDL", "LAMA", "EMD*", "K-Motif"]

    plot_motifsets(
        ds_name,
        df,
        motifsets=motifs,
        motiflet_dims=dims,
        motifset_names=motifset_names,
        motif_length=m,
        ground_truth=ground_truth,
        show=path is None)

    if path is not None:
        plt.savefig(path)
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
