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


def test_audio():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6']
    df = df.loc[channels]

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

    # if wave is present, extract audio snippets
    extract_motif_from_audio(df, index_range, length_in_seconds, ml, motif_length,
                             lrc_url, audio_file_url)


def test_publication():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

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

    # if wave is present, extract audio snippets
    extract_motif_from_audio(df, index_range, length_in_seconds, ml, motif_length,
                             lrc_url, audio_file_url)


def test_mstamp():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2',
                'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    run_mstamp(df, ds_name, motif_length=232)


def test_plot_both():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2',
                'MFCC 3', 'MFCC 4', 'MFCC 5',
                'MFCC 6']
    df = df.loc[channels]

    motif_length = 232

    path = "images_paper/audio/" + ds_name + ".pdf"

    motifs = [  # mstamp
        [9317, 9382],
        # motiflets
        [1146, 1408, 2183, 2442, 3217,
         3477, 4276, 4535, 5312, 5572],
        [5954, 6083, 6467, 6596, 6790,
         7081, 7210, 7519, 8018, 8147,
         8277, 8440, 8733, 8895]
    ]

    dims = [  # mstamp
        [0],
        # motiflets
        [0, 1, 2],
        [0, 1, 5]
    ]

    motifset_names = ["mStamp + MDL", "1st Motiflets", "2nd Motiflets"]

    plot_motifsets(
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
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[:, 0], data[:, 1]

            ax[i].specgram(left, Fs=samplingFrequency, cmap='plasma')
            ax[i].set_ylabel("Freq.")

            ax[i].set_ylim([0, 5000])
            # ax[i].set_xlim([0, 0.92])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
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
