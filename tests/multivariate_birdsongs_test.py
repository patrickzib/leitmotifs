import scipy.cluster.hierarchy as sch
from audio.lyrics import *
from pydub import AudioSegment

import audioread
import librosa

# import matplotlib
# matplotlib.use('macosx')

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/birds/"

datasets = {
    "Common-Starling": {
        "ks": 5,
        "channels": 10,
        "length_range": np.arange(25, 100, 1),
        "ds_name": "Common-Starling",
        "audio_file_url": path + "xc27154---common-starling---sturnus-vulgaris.mp3",
    },
    "House-Sparrow": {
        "ks": 10,
        "channels": 10,
        "length_range": np.arange(25, 50, 1),
        "ds_name": "House-Sparrow",
        "audio_file_url": path + "house-sparrow-passer-domesticus-audio.mp3"
    }
}

dataset = datasets["House-Sparrow"]
k_max = dataset["ks"]
channels = dataset["channels"]
length_range = dataset["length_range"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]


def read_bird_song():
    # Read audio from wav file
    aro = audioread.ffdec.FFmpegAudioFile(audio_file_url)
    x, sr = librosa.load(aro, mono=True)
    mfcc_f = librosa.feature.mfcc(y=x, sr=sr)
    audio_length_seconds = librosa.get_duration(filename=audio_file_url)
    print("Length:", sr, "in seconds", audio_length_seconds, "s")
    index_range = np.arange(0, mfcc_f.shape[1]) * audio_length_seconds / mfcc_f.shape[1]
    df = pd.DataFrame(mfcc_f,
                      index=["MFCC " + str(a) for a in np.arange(0, mfcc_f.shape[0])],
                      columns=index_range)
    df.index.name = "MFCC"
    return df, index_range


def _extract_audio_segment(channels, index_range, length_in_seconds, motiflet):
    song = AudioSegment.from_mp3(audio_file_url)
    for a, motif in enumerate(motiflet):
        start = (index_range[motif]) * 1000  # ms
        end = start + length_in_seconds * 1000  # ms
        motif_audio = song[start:end]
        motif_audio.export('audio/bird_songs/' + ds_name +
                           "_Channel_" + str(channels) +
                           "_Motif_" + str(a) + '.wav', format="wav")


def test_audio():
    # channels = ['MFCC 1', 'MFCC 2']
    channels = ['MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9']
    df, index_range = read_bird_song()
    df = df.loc[channels]

    ml = Motiflets(ds_name, df,
                   slack=1.0,
                   dimension_labels=df.index
                   )

    motif_length, all_minima = ml.fit_motif_length(k_max, length_range)
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    # length_in_seconds = 2.2
    # motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    # print(motif_length)

    dists, motiflets, elbow_points = ml.fit_k_elbow(
        k_max, motif_length=motif_length,
        plot_elbows=True,
        plot_motifs_as_grid=False
    )

    path_ = ("audio/bird_songs/" + ds_name +
             "_Channels_" + str(len(df.index)) +
             "_full.pdf")
    ml.plot_dataset(path_)

    # best motiflet
    motiflet = np.sort(motiflets[elbow_points[-1]])
    print("Positions:", index_range[motiflet])

    ml.plot_motifset()

    # plot_motifset(
    #     ds_name,
    #     df.iloc[:, : min(np.max(motiflet) + 2 * motif_length, df.shape[1])],
    #     motifset=motiflet,
    #     dist=dists[elbow_points[0]],
    #     motif_length=motif_length,
    #     show=False)

    plt.savefig(
        "audio/bird_songs/" + ds_name + "_Channels_" + str(
            len(df.index)) + "_Motif.pdf")
    plt.show()

    _extract_audio_segment(channels, index_range, length_in_seconds, motiflet)


def test_dendrogram():
    df, index_range = read_bird_song()
    df = df.iloc[0:channels]

    motif_length, all_minima = plot_motif_length_selection(
        k_max,
        df,
        length_range,
        ds_name,
        slack=1.0
    )

    # motif_length = 25
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    ml = Motiflets(ds_name, df,
                   elbow_deviation=1.25,
                   slack=1.0,
                   dimension_labels=df.index
                   )

    ml.fit_dendrogram(k_max, motif_length, n_clusters=4)
