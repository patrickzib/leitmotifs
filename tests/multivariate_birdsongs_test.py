import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 150

from audio.lyrics import *

path = "../../motiflets_use_cases/birds/"

datasets = {
    "Common-Starling": {
        "ks": 5,
        "channels": 10,
        "length_range": np.arange(25, 100, 5),
        "ds_name": "Common-Starling",
        "audio_file_url": path + "xc27154---common-starling---sturnus-vulgaris.mp3",
    },
    "House-Sparrow": {
        "ks": 20,
        "channels": 10,
        "length_range": np.arange(25, 50, 5),
        "ds_name": "House-Sparrow",
        "audio_file_url": path + "house-sparrow-passer-domesticus-audio.mp3"
    }
}

dataset = datasets["Common-Starling"]
k_max = dataset["ks"]
channels = dataset["channels"]
length_range = dataset["length_range"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]


def test_audio():
    seconds, df, index_range = read_mp3(audio_file_url)

    ml = Motiflets(ds_name,
                   df.iloc[:channels, :],
                   slack=1.0,
                   dimension_labels=df.index,
                   n_dims=2,
                   )

    motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range,
        plot_motifsets=False
    )
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    path_ = ("audio/bird_songs/" + ds_name +
             "_Channels_" + str(len(df.index)) +
             "_full.pdf")
    ml.plot_motifset()

    plt.savefig(
        "audio/bird_songs/" + ds_name + "_Channels_" + str(
            len(df.index)) + "_Motif.pdf")
    plt.show()

    extract_audio_segment(
        df, ds_name, audio_file_url, "bird_songs",
        length_in_seconds, index_range, motif_length, ml.motiflets[ml.elbow_points[-1]])


def test_publication():
    seconds, df, index_range = read_mp3(audio_file_url)

    ml = Motiflets(ds_name,
                   df.iloc[:channels, :],
                   slack=1.0,
                   dimension_labels=df.index,
                   n_dims=2,
                   )

    motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range,
        plot_motifsets=False
    )
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    path_ = ("images_paper/bird_songs/" + ds_name + ".pdf")
    ml.plot_motifset(path=path_)

    extract_audio_segment(
        df, ds_name, audio_file_url, "bird_songs",
        length_in_seconds, index_range, motif_length, ml.motiflets[ml.elbow_points[-1]])

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:")
        print("\tpos\t:", motiflet)
        print("\tdims\t:", ml.motiflets_dims[eb])


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)

    offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        samplingFrequency, data = read_wave(audio_file_url)
        left, right = data[offset[i]:, 0], data[offset[i]:, 1]

        ax[i].specgram(left, Fs=samplingFrequency, cmap='plasma')
        ax[i].set_ylabel("Freq.")

        ax[i].set_ylim([0, 10000])
        ax[i].set_xlim([0, 0.92])

    # for a in ax:
    # a.set_xticklabels([])
    # a.set_yticklabels([])

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0.1)

    plt.savefig("images_paper/bird_songs/spectrogram.pdf")


def test_plot_spectrogram():
    audio_file_urls = \
        ["images_paper/bird_songs/Common-Starling_Dims_20_Length_50_Motif_0.wav",
         "images_paper/bird_songs/Common-Starling_Dims_20_Length_50_Motif_1.wav",
         "images_paper/bird_songs/Common-Starling_Dims_20_Length_50_Motif_2.wav",
         "images_paper/bird_songs/Common-Starling_Dims_20_Length_50_Motif_3.wav"]

    plot_spectrogram(audio_file_urls)
    plt.show()


def test_mstamp():
    import stumpy

    seconds, df, index_range = read_mp3(audio_file_url)
    series = df.values.astype(np.float64)
    m = 50  # As used by k-Motiflets

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

    # extract_audio_segment(
    #    df, ds_name, audio_file_url, "snippets",
    #    length_in_seconds, index_range, m, motif, id=1)


def test_plot_both():
    seconds, df, index_range = read_mp3(audio_file_url)

    motif_length = 50

    path = "images_paper/bird_songs/" + ds_name + ".pdf"

    motifs = [  # mstamp
        [2135, 2227],
        # motiflets
        [509, 566, 2137, 2229]
    ]

    dims = [  # mstamp
        [1],
        # motiflets
        [1, 3]
    ]

    motifset_names = ["mStamp + MDL", "1st Motiflets"]

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

    plt.savefig("images_paper/bird-songs/spectrogram.pdf")
