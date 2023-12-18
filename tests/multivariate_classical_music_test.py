import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

path = "../../motiflets_use_cases/audio/"

datasets = {
    "Hans Zimmer - He is a Pirate": {
        "ks": 10,
        "channels": 5,
        "length_in_seconds": np.arange(1.0, 5, 0.25),
        "ds_name": "Hans Zimmer - He is a Pirate",
        "audio_file_url": path + "Hans Zimmer - He is a Pirate.mp3",
    },
    "Hans Zimmer - Zoosters Breakout": {
        "ks": 10,
        "channels": 5,
        "length_in_seconds": np.arange(4.0, 6.0, 0.5),
        "ds_name": "Hans Zimmer - Zoosters Breakout",
        "audio_file_url": path + "Hans Zimmer - Zoosters Breakout.mp3",
    },
    "Lord of the Rings Symphony - The Shire": {
        "ks": 8,
        "channels": 5,
        "length_in_seconds": np.arange(7.0, 8.0, 0.25),
        "ds_name": "Lord of the Rings Symphony - The Shire",
        "audio_file_url": path + "Lord of the Rings Symphony - The Shire.mp3",
    },
}

dataset = datasets["Lord of the Rings Symphony - The Shire"]
k_max = dataset["ks"]
n_dims = dataset["channels"]
motif_length_range_in_s = dataset["length_in_seconds"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]


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


def test_stumpy():
    import stumpy
    m = 301

    audio_length_seconds, df, index_range = read_songs()
    channels = [  # 'MFCC 0',
        'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
        'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
        'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15',
        # 'MFCC 16', 'MFCC 17', 'MFCC 18', 'MFCC 19'
    ]
    data = df.loc[channels].astype(np.float64).values
    mps, indices = stumpy.mstump(data, m)
    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    pos = motifs_idx[n_dims - 1]
    pos2 = nn_idx[n_dims - 1]

    print("Positions:", index_range[pos], index_range[pos2])


def test_publication():
    audio_length_seconds, df, index_range = read_songs()
    channels = [  # 'MFCC 0',
        'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
        'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
        'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15',
        # 'MFCC 16', 'MFCC 17', 'MFCC 18', 'MFCC 19'
    ]
    df = df.loc[channels]

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   slack=1.0,
                   )

    motif_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=False,
        plot_motifsets=False,
        plot_best_only=True,
    )
    ml.plot_motifset(path="images_paper/audio/" + ds_name + ".pdf")

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    # best motiflets
    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:", index_range[motiflet])

        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets",
            length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)
    ax[0].set_title("Spectrogram of found Leitmotif", size=16)

    # offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        samplingFrequency, data = read_wave(audio_file_url)
        left, right = data[:, 0], data[:, 1]

        ax[i].specgram(left, Fs=samplingFrequency, cmap='Spectral')
        ax[i].set_ylabel("Freq.")

        ax[i].set_ylim([0, 5000])
        # ax[i].set_xlim([0, 0.92])

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0.1)

    plt.savefig("images_paper/audio/lotr-spectrogram.pdf")


def test_plot_spectrogram():
    audio_file_urls = \
        [
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_0.wav",
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_1.wav",
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_2.wav",
            "audio/snippets/Lord of the Rings Symphony - The Shire_Dims_15_Length_301_Motif_1_3.wav"
        ]
    #        [
    #            "audio/snippets/Hans Zimmer - Zoosters Breakout_Dims_14_Length_172_Motif_1_0.wav",
    #            "audio/snippets/Hans Zimmer - Zoosters Breakout_Dims_14_Length_172_Motif_1_1.wav",
    #            "audio/snippets/Hans Zimmer - Zoosters Breakout_Dims_14_Length_172_Motif_1_2.wav",
    #            "audio/snippets/Hans Zimmer - Zoosters Breakout_Dims_14_Length_172_Motif_1_3.wav",
    #            "audio/snippets/Hans Zimmer - Zoosters Breakout_Dims_14_Length_172_Motif_1_4.wav",
    #            "audio/snippets/Hans Zimmer - Zoosters Breakout_Dims_14_Length_172_Motif_1_5.wav"]

    plot_spectrogram(audio_file_urls)
    plt.show()
