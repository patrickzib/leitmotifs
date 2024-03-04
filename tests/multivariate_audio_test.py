import matplotlib as mpl
import os

from audio.lyrics import *
from motiflets.competitors import *
from motiflets.motiflets import read_audio_from_dataframe

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

datasets = {
    "Numb - Linkin Park": {
        "ks": [5],
        "n_dims": 5,
        "motif_length": 226,
        "length_range_in_seconds": np.arange(5.0, 6.0, 0.25),
        "slack": 1.0,
        "ds_name": "Numb - Linkin Park",
        "audio_file_url": path_to_wav + "Numb - Linkin Park.wav",
        "pandas_file_url": path + "Numb-Linkin-Park.csv",
        "lrc_url": path_to_wav + "Numb - Linkin Park.lrc",
    },
    "What I've Done - Linkin Park": {
        "ks": [6],
        "n_dims": 3,
        "motif_length": 300,
        "length_range_in_seconds": np.arange(19.0, 21, 0.25),
        "slack": 1.0,
        "ds_name": "What I've Done - Linkin Park",
        "audio_file_url": path_to_wav + "What I've Done - Linkin Park.wav",
        "pandas_file_url": path + "What-I-ve-Done-Linkin-Park.csv",
        "lrc_url": path_to_wav + "What I've Done - Linkin Park.lrc"
    },
    "The Rolling Stones - Paint It, Black": {
        "ks": [10, 14],
        "n_dims": 3,
        "motif_length": 232,
        "length_range_in_seconds": np.arange(5.0, 6.0, 0.1),
        "slack": 1.0,
        "ds_name": "The Rolling Stones - Paint It, Black",
        "audio_file_url": path_to_wav + "The Rolling Stones - Paint It, Black.wav",
        "pandas_file_url": path + "The-Rolling-Stones-Paint-It-Black.csv",
        "lrc_url": path_to_wav + "The Rolling Stones - Paint It, Black.lrc"
    },
    "Vanilla Ice - Ice Ice Baby": {
        "ks": [20],
        "n_dims": 6,
        "motif_length": 180,
        "length_range_in_seconds": np.arange(170, 190, 10),
        "slack": 0.6,
        "ds_name": "Vanilla Ice - Ice Ice Baby",
        "audio_file_url": path_to_wav + "Vanilla_Ice-Ice_Ice_Baby.mp3",
        "pandas_file_url": path + "Vanilla_Ice-Ice_Ice_Baby.csv",
        "lrc_url": None
    },
    "Queen David Bowie - Under Pressure": {
        "ks": [16],
        "n_dims": 5,
        "motif_length": 180,
        "length_range_in_seconds": np.arange(170, 190, 10),
        "slack": 0.6,
        "ds_name": "Queen David Bowie - Under Pressure",
        "audio_file_url": path_to_wav + "Queen-David-Bowie-Under-Pressure.mp3",
        "pandas_file_url": path + "Queen-David-Bowie-Under-Pressure.csv",
        "lrc_url": None
    }
}

# dataset = datasets["The Rolling Stones - Paint It, Black"]
# dataset = datasets["What I've Done - Linkin Park"]
# dataset = datasets["Numb - Linkin Park"]
# dataset = datasets["Vanilla Ice - Ice Ice Baby"]
dataset = datasets["Queen David Bowie - Under Pressure"]

ks = dataset["ks"]
n_dims = dataset["n_dims"]
slack = dataset["slack"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
pandas_file_url = dataset["pandas_file_url"]
lrc_url = dataset["lrc_url"]
m = dataset["motif_length"]

# for learning parameters
k_max = np.max(dataset["ks"]) + 2
motif_length_range_in_s = dataset["length_range_in_seconds"]

channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4',
            'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10']


# def test_read_write():
# audio_length_seconds, df, index_range = read_from_wav(read_from_wav)
# df.to_csv(pandas_file_url, compression='gzip')
# audio_length_seconds2, df2, index_range2 = read_from_dataframe(pandas_file_url)


def extract_motif_from_audio(
        df, index_range, length_in_seconds, ml, motif_length,
        lrc_url, audio_file_url
):
    if os.path.isfile(lrc_url) and os.path.isfile(audio_file_url):
        subtitles = read_lrc(lrc_url)
        df_sub = get_dataframe_from_subtitle_object(subtitles)
        df_sub.set_index("seconds", inplace=True)

        # best motiflets
        for a, eb in enumerate(ml.elbow_points):
            motiflet = np.sort(ml.motiflets[eb])
            print("Positions:", index_range[motiflet])

            lyrics = []
            for i, m in enumerate(motiflet):
                l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
                lyrics.append(l)
                print(i + 1, l)

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))

    else:
        print("No lyrics or audio file found.")


def test_ground_truth():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    df = df.loc[channels]

    # motif_length_range = np.int32(motif_length_range_in_s /
    #                              audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth
                   )

    # print("Positions:", index_range[ground_truth.loc[0][0]])

    positions = []
    # [24.50, 40.5], [68.5, 84.50], [108.5, 124.5], [162.5, 178.5], [178.5, 194.5]
    positions.append([[index_range.searchsorted(24.5), index_range.searchsorted(40.5)],
                      [index_range.searchsorted(68.5), index_range.searchsorted(84.50)],
                      [index_range.searchsorted(108.5),
                       index_range.searchsorted(124.5)],
                      [index_range.searchsorted(162.5),
                       index_range.searchsorted(178.5)],
                      [index_range.searchsorted(178.5), index_range.searchsorted(194.5)]
                      ])
    print(positions)

    if os.path.isfile(audio_file_url):
        # extract motiflets
        for a, motif in enumerate(ground_truth.loc[0]):
            motif_length = motif[a][1] - motif[a][0]
            length_in_seconds = motif_length * audio_length_seconds / df.shape[1]

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length,
                np.array(motif)[:, 0], id=(a + 1))

    ml.plot_dataset()


def test_learn_parameters():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    n_dims = 3
    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth,
                   slack=slack
                   )

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])
    m, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=False,
        plot_motifsets=True,
        plot_best_only=False
    )

    ml.plot_motifset(motifset_name="LAMA")

    length_in_seconds = m * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, m)

    # if wave is present, extract audio snippets
    extract_motif_from_audio(df, index_range, length_in_seconds, ml, m,
                             lrc_url, audio_file_url)


def test_publication():
    test_lama()
    test_emd_pca()
    test_mstamp()
    test_kmotifs()


def test_lama(use_PCA=False):
    # from sklearn.decomposition import PCA
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels=channels)

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        df_transform = pca.fit_transform(df.T).T
    else:
        df_transform = df

    ml = Motiflets(
        ds_name, df_transform,
        dimension_labels=df.index,
        n_dims=n_dims,
        ground_truth=ground_truth,
        slack=slack
    )

    # motif_length_range = np.int32(motif_length_range_in_s /
    #                               audio_length_seconds * df.shape[1])
    # m, _ = ml.fit_motif_length(
    #     k_max,
    #     motif_length_range,
    #     plot=False,
    #     plot_elbows=True,
    #     plot_motifsets=True,
    #     plot_best_only=True
    # )

    ml.fit_k_elbow(
        k_max=k_max,
        motif_length=m,
        plot_elbows=False,
        plot_motifsets=False,
    )
    ml.plot_motifset(
        elbow_points=ks,
        path="images_paper/audio/" + ds_name + ".pdf",
        motifset_name="LAMA")

    length_in_seconds = m * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, m)

    print("Positions:")
    # for eb in ml.elbow_points:
    for eb in ks:
        motiflet = np.sort(ml.motiflets[eb])
        print("\tMotif pos\t:", repr(motiflet))
        if use_PCA:
            print("\tdims\t:", repr(np.argsort(pca.components_[:])[:, :n_dims]))
        else:
            print("\tdims\t:", repr(ml.motiflets_dims[eb]))

    if False:
        # if wave is present, extract audio snippets
        extract_motif_from_audio(
            df, index_range, length_in_seconds, ml, m,
            lrc_url, audio_file_url)


def test_emd_pca():
    test_lama(use_PCA=True)


def test_mstamp():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    run_mstamp(df, ds_name, motif_length=m, ground_truth=ground_truth)


def test_kmotifs():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    for target_k in ks:
        _ = run_kmotifs(
            df,
            ds_name,
            m,
            r_ranges=np.arange(10, 600, 5),
            use_dims=df.shape[0],
            target_k=target_k,
            ground_truth=ground_truth,
            slack=slack
        )


def test_plot_all():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    # TODO by dataset
    motif_length = 232

    path = "images_paper/audio/" + ds_name + "_new.pdf"

    motifs = [
        # mstamp
        [9317, 9382],
        # LAMA
        [1146, 1408, 2183, 2442, 3217,
         3477, 4276, 4535, 5312, 5572],
        [5954, 6083, 6467, 6596, 6790,
         7081, 7210, 7519, 8018, 8147,
         8277, 8440, 8733, 8895],
        # EMD*
        [663, 921, 1702, 1960, 2736,
         2994, 4829, 5087],
        [979, 5937, 6450, 6612, 7064, 7486,
         8001, 8147, 8260, 8439, 8781, 8927],
        # K-Motif
        [5848, 6074, 6360, 6588, 6878,
         7104, 7234, 7381, 7655, 7911,
         8171, 8398, 8691, 8951, 9212],

    ]

    dims = [
        # mstamp
        [0],
        # LAMA
        [0, 1, 2],
        [0, 1, 5],
        # EMD*
        [0, 1, 3],
        [0, 1, 3],
        # K-Motif
        np.arange(10),
        np.arange(10)
    ]

    motifset_names = ["mStamp + MDL",
                      "1st LAMA", "2nd LAMA",
                      "1st EMD*", "2nd EMD*",
                      "1st K-Motif", "2nd K-Motif"]

    plot_motifsets(
        ds_name,
        df,
        motifsets=motifs,
        motiflet_dims=dims,
        motifset_names=motifset_names,
        # dist=self.dists[elbow_points],
        motif_length=motif_length,
        ground_truth=ground_truth,
        show=path is None)

    if path is not None:
        plt.savefig(path)
        plt.show()


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)

    # offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[:, 0], data[:, 1]

            ax[i].specgram(left,
                           Fs=samplingFrequency,  # cmap='Grays',
                           # scale='dB', # vmin=-50, vmax=0
                           )
            ax[i].set_ylabel("Freq.")

            ax[i].set_ylim([0, 5000])
            # ax[i].set_xlim([0, 0.92])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig("images_paper/audio/rolling-stones-spectrogram_new.pdf")


def test_plot_spectrogram():
    audio_file_urls = \
        [
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_0.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_1.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_2.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_3.wav"]

    plot_spectrogram(audio_file_urls)
    plt.show()
