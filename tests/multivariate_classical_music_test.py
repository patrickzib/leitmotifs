import os
import matplotlib as mpl
from motiflets.motiflets import read_audio_from_dataframe
from motiflets.motiflets import read_ground_truth

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

datasets = {
    "Hans Zimmer - He is a Pirate": {
        "ks": 10,
        "channels": 5,
        "length_in_seconds": np.arange(1.0, 5, 0.25),
        "ds_name": "Hans Zimmer - He is a Pirate",
        "audio_file_url": path_to_wav + "Hans Zimmer - He is a Pirate.mp3",
        "pandas_file_url": path + "Hans-Zimmer-He-is-a-Pirate.csv",
    },
    "Hans Zimmer - Zoosters Breakout": {
        "ks": 10,
        "channels": 5,
        "length_in_seconds": np.arange(4.0, 6.0, 0.5),
        "ds_name": "Hans Zimmer - Zoosters Breakout",
        "audio_file_url": path_to_wav + "Hans Zimmer - Zoosters Breakout.mp3",
        "pandas_file_url": path + "Hans-Zimmer-Zoosters-Breakout.csv",
    },
    "Lord of the Rings Symphony - The Shire": {
        "ks": 8,
        "channels": 5,
        "length_in_seconds": np.arange(7.0, 8.0, 0.25),
        "ds_name": "Lord of the Rings Symphony - The Shire",
        "audio_file_url": path_to_wav + "Lord of the Rings Symphony - The Shire.mp3",
        "pandas_file_url": path + "Lord-of-the-Rings-Symphony-The-Shire.csv",
    },
}

dataset = datasets["Lord of the Rings Symphony - The Shire"]
# dataset = datasets["Hans Zimmer - Zoosters Breakout"]
# dataset = datasets["Hans Zimmer - He is a Pirate"]
k_max = dataset["ks"]
n_dims = dataset["channels"]
motif_length_range_in_s = dataset["length_in_seconds"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
pandas_file_url = dataset["pandas_file_url"]

# def test_read_write():
#    audio_length_seconds, df, index_range = read_mp3(audio_file_url)
#    df.to_csv(pandas_file_url, compression='gzip')
#    audio_length_seconds2, df2, index_range2 = read_from_dataframe(pandas_file_url)


# def test_ground_truth():
#     audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
#
#     ground_truth = read_ground_truth(pandas_file_url, path="")
#
#     channels = [  # 'MFCC 0',
#         'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
#         'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
#         'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15'
#     ]
#     df = df.loc[channels]
#
#     ml = Motiflets(ds_name, df,
#                    dimension_labels=df.index,
#                    n_dims=n_dims,
#                    slack=1.0,
#                    ground_truth=ground_truth
#                    )
#
#     print("Positions:", index_range[ground_truth.loc[0][0]])
#
#     if os.path.isfile(audio_file_url):
#         # extract motiflets
#         for a, motif in enumerate(ground_truth.loc[0]):
#             motif_length = motif[0][1]-motif[0][0]
#             length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
#
#             extract_audio_segment(
#                 df, ds_name, audio_file_url, "snippets",
#                 length_in_seconds, index_range, motif_length,
#                 np.array(motif)[:,0], id=(a + 1))
#
#     ml.plot_dataset()

def test_publication():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)

    ground_truth = read_ground_truth(pandas_file_url, path="")

    channels = [  # 'MFCC 0',
        'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
        'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
        'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15'
    ]
    df = df.loc[channels]

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   slack=1.0,
                   ground_truth=ground_truth
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

    if os.path.isfile(audio_file_url):
        # extract motiflets
        for a, eb in enumerate(ml.elbow_points):
            motiflet = np.sort(ml.motiflets[eb])
            print("Positions:", index_range[motiflet])
            print("Positions:", list(zip(motiflet, motiflet+motif_length)))

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)
    ax[0].set_title("Spectrogram of found Leitmotif", size=16)

    for i, audio_file_url in enumerate(audio_file_urls):
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[:, 0], data[:, 1]
            ax[i].specgram(left, Fs=samplingFrequency, cmap='Spectral')
            ax[i].set_ylabel("Freq.")
            ax[i].set_ylim([0, 5000])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig("images_paper/audio/lotr-spectrogram.pdf")


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
