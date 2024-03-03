import matplotlib as mpl
import os

from audio.lyrics import *
from motiflets.competitors import *
from motiflets.motiflets import read_audio_from_dataframe, read_ground_truth

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

datasets = {
    "Numb - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": np.arange(2.5, 4.5, 0.1),
        "ds_name": "Numb - Linkin Park",
        "audio_file_url": path_to_wav + "Numb - Linkin Park.wav",
        "pandas_file_url": path + "Numb-Linkin-Park.csv",
        "lrc_url": path_to_wav + "Numb - Linkin Park.lrc"
    },
    "What I've Done - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": np.arange(6.0, 8, 0.1),  # 7.75,
        "ds_name": "What I've Done - Linkin Park",
        "audio_file_url": path_to_wav + "What I've Done - Linkin Park.wav",
        "pandas_file_url": path + "What-I-ve-Done-Linkin-Park.csv",
        "lrc_url": path_to_wav + "What I've Done - Linkin Park.lrc"
    },
    "The Rolling Stones - Paint It, Black": {
        "ks": 15,
        "channels": 3,
        "length_in_seconds": np.arange(5.0, 6.0, 0.1),
        "ds_name": "The Rolling Stones - Paint It, Black",
        "audio_file_url": path_to_wav + "The Rolling Stones - Paint It, Black.wav",
        "pandas_file_url": path + "The-Rolling-Stones-Paint-It-Black.csv",
        "lrc_url": path_to_wav + "The Rolling Stones - Paint It, Black.lrc"
    }
}

dataset = datasets["The Rolling Stones - Paint It, Black"]
# dataset = datasets["What I've Done - Linkin Park"]
# dataset = datasets["Numb - Linkin Park"]

k_max = dataset["ks"]
n_dims = dataset["channels"]
motif_length_range_in_s = dataset["length_in_seconds"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
pandas_file_url = dataset["pandas_file_url"]
lrc_url = dataset["lrc_url"]


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
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    ground_truth = read_ground_truth(pandas_file_url, path="")

    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6']
    df = df.loc[channels]

    # motif_length_range = np.int32(motif_length_range_in_s /
    #                              audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth
                   )

    print("Positions:", index_range[ground_truth.loc[0][0]])

    # positions = []
    # # B1 - I see a red door and I want it painted black
    # positions.append([[index_range.searchsorted(14.8), index_range.searchsorted(19.90)],
    #                   [index_range.searchsorted(20.70), index_range.searchsorted(26)]])
    #
    # # C1 - I see the girls walk by, dressed in their summer clothes
    # positions.append([[index_range.searchsorted(26.70), index_range.searchsorted(32.10)],
    #                   [index_range.searchsorted(32.80), index_range.searchsorted(37.90)]])
    #
    # # B2 - I see a line of cars and they're all painted black
    # positions.append([[index_range.searchsorted(38.90), index_range.searchsorted(44.20)],
    #                   [index_range.searchsorted(44.90), index_range.searchsorted(50.10)]])
    #
    # # C2 - I see people turn their heads and quickly look away
    # positions.append([[index_range.searchsorted(50.80), index_range.searchsorted(55.70)],
    #                   [index_range.searchsorted(56.80), index_range.searchsorted(61.80)]])
    #
    # # B3 - I look inside myself and see my heart is black
    # positions.append([[index_range.searchsorted(62.90), index_range.searchsorted(68.00)],
    #                   [index_range.searchsorted(68.80), index_range.searchsorted(73.90)]])
    #
    # # C3 - Maybe then I'll fade away and not have to face the facts
    # positions.append([[index_range.searchsorted(74.80), index_range.searchsorted(79.70)],
    #                   [index_range.searchsorted(80.80), index_range.searchsorted(85.80)]])
    #
    # # D1 - No more will my green sea go turn a deeper blue
    # positions.append([[index_range.searchsorted(86.90), index_range.searchsorted(92.10)],
    #                   [index_range.searchsorted(93.30), index_range.searchsorted(98.50)]])
    #
    # # C4 - If I look hard enough into the setting sun
    # positions.append([[index_range.searchsorted(99.40), index_range.searchsorted(104.40)],
    #                   [index_range.searchsorted(105.40), index_range.searchsorted(110.60)]])
    #
    # # B1 - I see a red door and I want it painted black
    # positions.append([[index_range.searchsorted(112.00), index_range.searchsorted(116.40)],
    #                   [index_range.searchsorted(117.50), index_range.searchsorted(122.30)]])
    #
    # # C1 - I see the girls walk by, dressed in their summer clothes
    # positions.append([[index_range.searchsorted(123.40), index_range.searchsorted(128.70)],
    #                   [index_range.searchsorted(129.40), index_range.searchsorted(134.60)]])
    #
    # # F - Humming
    # positions.append([[index_range.searchsorted(135.50), index_range.searchsorted(158.20)],
    #                   [index_range.searchsorted(177.60), index_range.searchsorted(213.30)]])
    #
    # print(positions)

    if os.path.isfile(audio_file_url):
        # extract motiflets
        for a, motif in enumerate(ground_truth.loc[0]):
            motif_length = motif[0][1] - motif[0][0]
            length_in_seconds = motif_length * audio_length_seconds / df.shape[1]

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length,
                np.array(motif)[:, 0], id=(a + 1))

    ml.plot_dataset()


def test_audio():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    ground_truth = read_ground_truth(pandas_file_url, path="")
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6']
    df = df.loc[channels]

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth
                   )

    motif_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=True,
        plot_motifsets=True,
        plot_best_only=False
    )

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    # if wave is present, extract audio snippets
    extract_motif_from_audio(df, index_range, length_in_seconds, ml, motif_length,
                             lrc_url, audio_file_url)


def test_publication(use_PCA=False):
    # from sklearn.decomposition import PCA
    ground_truth = read_ground_truth(pandas_file_url, path="")

    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        df_transform = pca.fit_transform(df.T).T
    else:
        df_transform = df

    ml = Motiflets(ds_name, df_transform,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth
                   )

    motif_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=False,
        plot_elbows=True,
        plot_motifsets=True,
        plot_best_only=True
    )
    ml.plot_motifset(  # elbow_points=[10, 14],
        path="images_paper/audio/" + ds_name + ".pdf")

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    # if wave is present, extract audio snippets
    extract_motif_from_audio(df, index_range, length_in_seconds, ml, motif_length,
                             lrc_url, audio_file_url)

    print("Positions:")
    for eb in ml.elbow_points:
        motiflet = np.sort(ml.motiflets[eb])
        print("\tpos\t:", repr(motiflet))

        if use_PCA:
            print("\tdims\t:", repr(np.argsort(pca.components_[:])[:, :n_dims]))
        else:
            print("\tdims\t:", repr(ml.motiflets_dims[eb]))


def test_univariate_pca():
    test_publication(use_PCA=True)


def test_mstamp():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2',
                'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    m = 232
    run_mstamp(df, ds_name, motif_length=m)


def test_kmotifs():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2',
                'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    for target_k in [10, 14]:
        m = 232
        _ = run_kmotifs(
            df,
            ds_name,
            m,
            r_ranges=np.arange(100, 300, 5),
            use_dims=8,
            target_k=target_k,
        )


def test_plot_all():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    ground_truth = read_ground_truth(pandas_file_url, path="")

    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2',
                'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

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
