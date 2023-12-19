import os
import matplotlib as mpl
from motiflets.motiflets import read_audio_from_dataframe

mpl.rcParams['figure.dpi'] = 150

from audio.lyrics import *
from motiflets.competitors import *

# path outside the git
path_to_wav = "../../motiflets_use_cases/birds/"
path = "../datasets/audio/"

datasets = {
    "Common-Starling": {
        "ks": 5,
        "channels": 10,
        "length_range": np.arange(25, 100, 5),
        "ds_name": "Common-Starling",
        "audio_file_url": path_to_wav + "xc27154---common-starling---sturnus-vulgaris.mp3",
        "pandas_file_url": path + "common-starling-sturnus-vulgaris.csv"
    },
    "House-Sparrow": {
        "ks": 20,
        "channels": 10,
        "length_range": np.arange(25, 50, 5),
        "ds_name": "House-Sparrow",
        "audio_file_url": path_to_wav + "house-sparrow-passer-domesticus-audio.mp3",
        "pandas_file_url": path + "house-sparrow-passer-domesticus.csv"
    }
}

dataset = datasets["Common-Starling"]
# dataset = datasets["House-Sparrow"]
k_max = dataset["ks"]
channels = dataset["channels"]
length_range = dataset["length_range"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
pandas_file_url = dataset["pandas_file_url"]


# def test_read_write():
#    audio_length_seconds, df, index_range = read_from_wav(audio_file_url)
#    df.to_csv(pandas_file_url, compression='gzip')
#    audio_length_seconds2, df2, index_range2 = read_from_dataframe()


def test_publication():
    seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)

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

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:")
        print("\tpos\t:", motiflet)
        print("\tdims\t:", ml.motiflets_dims[eb])

    if os.path.isfile(audio_file_url):
        extract_audio_segment(
            df, ds_name, audio_file_url, "bird_songs",
            length_in_seconds, index_range, motif_length,
            ml.motiflets[ml.elbow_points[-1]])


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)

    offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[offset[i]:, 0], data[offset[i]:, 1]

            ax[i].specgram(left, Fs=samplingFrequency, cmap='plasma')
            ax[i].set_ylabel("Freq.")

            ax[i].set_ylim([0, 10000])
            ax[i].set_xlim([0, 0.92])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
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
    seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    m = 50  # As used by k-Motiflets
    run_mstamp(df, ds_name, motif_length=m)

    # extract_audio_segment(
    #    df, ds_name, audio_file_url, "snippets",
    #    length_in_seconds, index_range, m, motif, id=1)


def test_plot_both():
    seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)

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

    plot_motifsets(
        ds_name,
        df,
        motifsets=motifs,
        motiflet_dims=dims,
        motifset_names=motifset_names,
        motif_length=motif_length,
        show=path is None)

    if path is not None:
        plt.savefig(path)
        plt.show()
