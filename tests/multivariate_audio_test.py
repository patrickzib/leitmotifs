import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

path = "../../motiflets_use_cases/audio/"

datasets = {
    "Numb - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": np.arange(2.5, 4.5, 0.1),
        "ds_name": "Numb - Linkin Park",
        "audio_file_url": path + "Numb - Linkin Park.wav",
        "lrc_url": path + "Numb - Linkin Park.lrc"
    },
    "What I've Done - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": np.arange(6.0, 8, 0.1),  # 7.75,
        "ds_name": "What I've Done - Linkin Park",
        "audio_file_url": path + "What I've Done - Linkin Park.wav",
        "lrc_url": path + "What I've Done - Linkin Park.lrc"
    },
    "The Rolling Stones - Paint It, Black": {
        "ks": 15,
        "channels": 3,
        "length_in_seconds": np.arange(5.0, 6.0, 0.1),
        "ds_name": "The Rolling Stones - Paint It, Black",
        "audio_file_url": path + "The Rolling Stones - Paint It, Black.wav",
        "lrc_url": path + "The Rolling Stones - Paint It, Black.lrc"
    }
}

dataset = datasets["The Rolling Stones - Paint It, Black"]
# dataset = datasets["What I've Done - Linkin Park"]
k_max = dataset["ks"]
n_dims = dataset["channels"]
motif_length_range_in_s = dataset["length_in_seconds"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
lrc_url = dataset["lrc_url"]


def read_songs():
    # Read audio from wav file
    x, sr, mfcc_f, audio_length_seconds = read_audio(audio_file_url, True)
    print("Length:", audio_length_seconds)
    index_range = np.arange(0, mfcc_f.shape[1]) * audio_length_seconds / mfcc_f.shape[1]
    df = pd.DataFrame(mfcc_f,
                      index=["MFCC " + str(a) for a in np.arange(0, mfcc_f.shape[0])],
                      columns=index_range
                      )
    df.index.name = "MFCC"
    return audio_length_seconds, df, index_range


def test_audio():
    audio_length_seconds, df, index_range = read_songs()
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6']
    df = df.loc[channels]

    subtitles = read_lrc(lrc_url)
    df_sub = get_dataframe_from_subtitle_object(subtitles)
    df_sub.set_index("seconds", inplace=True)

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
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

    # best motiflets
    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:", index_range[motiflet])

        lyrics = []
        for i, m in enumerate(motiflet):
            l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
            lyrics.append(l)
            print(i + 1, l)

        # path_ = "audio/snippets/" + ds_name + "_Dims_" + str(n_dims) + "_Motif.pdf"
        # ml.plot_motifset(path=path_)

        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets",
            length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))


def test_publication():
    audio_length_seconds, df, index_range = read_songs()
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    subtitles = read_lrc(lrc_url)
    df_sub = get_dataframe_from_subtitle_object(subtitles)
    df_sub.set_index("seconds", inplace=True)

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   )

    motif_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=False,
        plot_elbows=True,
        plot_motifsets=True,
        plot_best_only=True
    )
    ml.plot_motifset(elbow_points=[10, 14],
                     path="images_paper/audio/" + ds_name + ".pdf")

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    # best motiflets
    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:")
        print("\tpos\t:", motiflet)
        print("\tdims\t:", ml.motiflets_dims[eb])
        # print("\t", index_range[motiflet])

        lyrics = []
        for i, m in enumerate(motiflet):
            l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
            lyrics.append(l)
            print(i + 1, l)

        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets",
            length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))


def test_mstamp():
    import stumpy

    audio_length_seconds, df, index_range = read_songs()
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3',
                'MFCC 4', 'MFCC 5', 'MFCC 6']
    df = df.loc[channels]
    series = df.values.astype(np.float64)

    m = 232     # As used by k-Motiflets
    length_in_seconds = m * audio_length_seconds / df.shape[1]
    print("Used motif length", length_in_seconds, m)

    # Find the Pair Motif
    mps, indices = stumpy.mstump(series, m=m)
    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    # Find the optimal dimensionality by minimizing the MDL
    mdls, subspaces = stumpy.mdl(series, m, motifs_idx, nn_idx)
    k = np.argmin(mdls)

    plt.plot(np.arange(len(mdls)), mdls, c='red', linewidth='2')
    plt.xlabel('k (zero-based)')
    plt.ylabel('Bit Size')
    plt.xticks(range(mps.shape[0]))
    plt.tight_layout()
    plt.show()

    print("Best dimensions", df.index[subspaces[k]])

    # found Pair Motif
    motif = [motifs_idx[subspaces[k]], nn_idx[subspaces[k]]]
    print("Pair Motif Position:")
    print("\tpos:\t", motif)
    print("\tf:  \t", subspaces[k])

    dims = [subspaces[k]]
    motifs = [[motifs_idx[subspaces[k]][0], nn_idx[subspaces[k]][0]]]
    motifset_names = ["mStamp"]

    fig, ax = plot_motifsets(
        ds_name,
        series,
        motifsets=motifs,
        motiflet_dims=dims,
        motifset_names=motifset_names,
        motif_length=m,
        show=True)



def test_plot_both():
    audio_length_seconds, df, index_range = read_songs()
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2',
                'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    motif_length = 232

    path = "images_paper/audio/" + ds_name + ".pdf"

    motifs = [# mstamp
              [9317, 9382],
              # motiflets
              [1146, 1408, 2183, 2442, 3217,
               3477, 4276, 4535, 5312, 5572],
              [5954, 6083, 6467, 6596, 6790,
               7081, 7210, 7519, 8018, 8147,
               8277, 8440, 8733, 8895]
              ]

    dims = [# mstamp
            [0],
            # motiflets
            [0, 1, 2],
            [0, 1, 5]
            ]

    motifset_names = ["mStamp + MDL", "1st Motiflets", "2nd Motiflets"]

    fig, ax = plot_motifsets(
        ds_name,
        df,
        motifsets=motifs,
        motiflet_dims=dims,
        motifset_names=motifset_names,
        # dist=self.dists[elbow_points],
        motif_length=motif_length,
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
        samplingFrequency, data = read_wave(audio_file_url)
        left, right = data[:, 0], data[:, 1]

        ax[i].specgram(left, Fs=samplingFrequency, cmap='plasma')
        ax[i].set_ylabel("Freq.")

        ax[i].set_ylim([0, 5000])
        # ax[i].set_xlim([0, 0.92])

    # for a in ax:
    # a.set_xticklabels([])
    # a.set_yticklabels([])

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0.1)

    plt.savefig("images_paper/audio/rolling-stones-spectrogram.pdf")


def test_plot_spectrogram():
    audio_file_urls = \
        [
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_0.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_1.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_2.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_3.wav"]

    plot_spectrogram(audio_file_urls)
    plt.show()
