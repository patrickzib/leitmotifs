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

    path_ = ("images_paper/bird_songs/" + ds_name +".pdf")
    ml.plot_motifset(path_)

    extract_audio_segment(
        df, ds_name, audio_file_url, "bird_songs",
        length_in_seconds, index_range, motif_length, ml.motiflets[ml.elbow_points[-1]])


def plot_spectrogram(audio_file_urls):

    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)

    offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        samplingFrequency, data = read_wave(audio_file_url)
        left, right = data[offset[i]:,0], data[offset[i]:,1]

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