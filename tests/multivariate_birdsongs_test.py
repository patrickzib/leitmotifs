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
ks = dataset["ks"]
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


def test_dendrogram():
    df, index_range = read_bird_song()
    df = df.iloc[0:channels]

    motif_length, all_minima = plot_motif_length_selection(
        ks,
        df,
        length_range,
        ds_name,
        slack=1.0
    )

    # motif_length = 25
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    dists, motiflets, elbow_points = plot_elbow_by_dimension(
        ks,
        df,
        dimension_labels=df.index,
        ds_name=ds_name,
        slack=1.0,
        motif_length=motif_length)

    series = np.zeros((df.shape[0], df.shape[1] - motif_length), dtype=np.float32)
    for i in range(series.shape[0]):
        for pos in motiflets[i, elbow_points[i][-1]]:
            series[i, pos:pos + motif_length] = 1

    X = series

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    Z = sch.linkage(X, method='ward')

    # creating the dendrogram
    _ = sch.dendrogram(
        Z, labels=df.index, ax=ax)

    ax.set_title('Dendrogram')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Euclidean distances')
    plt.tight_layout()
    plt.show()

    cluster_k = 4
    y_dimensions = sch.fcluster(Z, cluster_k, criterion='maxclust')
    mapping = list(zip(y_dimensions, df.index))

    joint_clusters = {}
    for i in range(1, cluster_k + 1):
        print("Cluster", i)
        joint_clusters[i] = [x[1] for x in mapping if x[0] == i]
        print(joint_clusters[i])
        print("----")


def test_audio():
    # channels = ['MFCC 1', 'MFCC 2']
    channels = ['MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9']
    df, index_range = read_bird_song()
    df = df.loc[channels]

    motif_length, all_minima = plot_motif_length_selection(
        ks,
        df,
        length_range,
        ds_name,
        slack=1.0
    )
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    # length_in_seconds = 2.2
    # motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    # print(motif_length)

    dists, motiflets, elbow_points = plot_elbow(
        ks,
        df,
        ds_name=ds_name,
        slack=1.0,
        plot_elbows=True,
        plot_grid=False,
        dimension_labels=df.index,
        motif_length=motif_length)

    plot_dataset(
        ds_name,
        df,
        show=False)

    plt.savefig(
       "audio/bird_songs/" + ds_name + "_Channels_" + str(len(df.index)) + "_full.pdf")
    plt.show()


    # dists, motiflets, elbow_points, motif_length = ml.search_k_motiflets_elbow(
    #    ks,
    #    df.values,
    #    motif_length=motif_length,
    #    # elbow_deviation=1.25,
    #    slack=1.0,
    # )

    # best motiflet
    motiflet = np.sort(motiflets[elbow_points[-1]])
    print("Positions:", index_range[motiflet])

    # plot_motiflet(
    #    df,
    #    motiflet,
    #    motif_length,
    #    title=ds_name
    # )
    # plt.tight_layout()
    # plt.savefig("audio/snippets/" + ds_name + "_Channels_" + str(len(df.index)) + "_Motif.pdf")
    # plt.show()

    plot_motifset(
        ds_name,
        df.iloc[:, : min(np.max(motiflet) + 2 * motif_length, df.shape[1])],
        motifset=motiflet,
        dist=dists[elbow_points[0]],
        motif_length=motif_length,
        show=False)

    plt.savefig(
       "audio/bird_songs/" + ds_name + "_Channels_" + str(len(df.index)) + "_Motif.pdf")
    plt.show()

    song = AudioSegment.from_mp3(audio_file_url)
    for a, motif in enumerate(motiflet):
        start = (index_range[motif]) * 1000  # ms
        end = start + length_in_seconds * 1000  # ms
        motif_audio = song[start:end]
        motif_audio.export('audio/bird_songs/' + ds_name +
                           "_Channel_" + str(channels) +
                           "_Motif_" + str(a) + '.wav', format="wav")
