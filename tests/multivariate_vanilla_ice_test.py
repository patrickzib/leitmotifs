import audioread
import librosa
import matplotlib as mpl
import scipy.cluster.hierarchy as sch
from pydub import AudioSegment

from audio.lyrics import *

# import matplotlib
# matplotlib.use('macosx')

mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/audio/"

A_dataset = "Queen-David-Bowie-Under-Pressure"
A_ds_name = "Queen David Bowie - Under Pressure"

B_ds_name = "Vanilla Ice - Ice Ice Baby"
B_dataset = "Vanilla_Ice-Ice_Ice_Baby"

datasets = [(A_dataset, A_ds_name), (B_dataset, B_ds_name)]

k_max = 25
length_in_seconds = 3.4  # in seconds


def read_mp3(audio_file_url):
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
    return audio_length_seconds, df, index_range


def test_dendrogram():
    for dataset, ds_name in datasets:

        audio_file_url = path + dataset + ".mp3"
        audio_length_seconds, df, index_range = read_mp3(audio_file_url)
        df = df.iloc[:10]

        # motif_length = plot_motif_length_selection(
        #    ks,
        #    df,
        #    np.arange(100, 200, 10),
        #    ds_name,
        # )
        motif_length = 200

        length_in_seconds = index_range[motif_length]
        print("Best length", motif_length, length_in_seconds, "s")

        dists, motiflets, elbow_points = plot_elbow_by_dimension(
            k_max,
            df,
            dimension_labels=df.index,
            ds_name=ds_name,
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
    for dataset, ds_name in datasets:
        audio_file_url = path + dataset + ".mp3"
        audio_length_seconds, df, index_range = read_mp3(audio_file_url)
        channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3']
        df = df.loc[channels]

        motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
        print(motif_length, length_in_seconds, "s")

        ml = Motiflets(ds_name, df,
                       # elbow_deviation=1.25,
                       slack=0.9,
                       dimension_labels=df.index
                       )

        dists, motiflets, elbow_points = ml.fit_k_elbow(
            k_max,
            plot_elbows=True,
            plot_motifs_as_grid=False,
            motif_length=motif_length)

        # best motiflet
        motiflet = np.sort(motiflets[elbow_points[-1]])
        print("Positions:", index_range[motiflet])

        path_ = "audio/snippets/" + ds_name + \
                "_Channels_" + str(len(df.index)) + \
                "_Motif.pdf"
        ml.plot_motifset(path_)

        song = AudioSegment.from_mp3(audio_file_url)
        for a, motif in enumerate(motiflet):
            start = (index_range[motif]) * 1000  # ms
            end = start + length_in_seconds * 1000  # ms
            motif_audio = song[start:end]
            motif_audio.export('audio/snippets/queen-vanilla-ice/' + ds_name +
                               # "_Channels_" + str(len(df.index)) +
                               "_Motif_" + str(a) + '.wav', format="wav")


def test_consensus():
    k_max = 40

    df_consensus = None
    audio_length_seconds = 0
    for dataset, ds_name in datasets:
        audio_file_url = path + dataset + ".mp3"
        seconds, df, index_range = read_mp3(audio_file_url)
        audio_length_seconds += seconds
        channels = ['MFCC 2', 'MFCC 3']
        df = df.loc[channels]

        if df_consensus is None:
            df_consensus = df
        else:
            df = pd.concat([df_consensus, df], axis=1)
            df.columns = np.arange(0, df.shape[1]) * audio_length_seconds / df.shape[1]

    ds_name = "Consensus"
    motif_length = 120  # int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length, length_in_seconds, "s")

    ml = Motiflets(ds_name, df,
                   # elbow_deviation=1.25,
                   slack=0.9,
                   dimension_labels=df.index,
                   )

    dists, motiflets, elbow_points = ml.fit_k_elbow(
        k_max,
        motif_length=motif_length,
        slack=0.9,
        plot_elbows=True,
        plot_grid=False)

    # best motiflet
    motiflet = np.sort(motiflets[elbow_points[-1]])
    print("Positions:", motiflet)

    path_ = "audio/snippets/queen-vanilla-ice/" + ds_name + "_Channels_" + str(
        len(df.index)) + "_Motif.pdf"
    ml.plot_motifset(path_)

    # song = AudioSegment.from_mp3(audio_file_url)
    # for a, motif in enumerate(motiflet):
    #    start = (index_range[motif]) * 1000  # ms
    #    end = start + length_in_seconds * 1000  # ms
    #    motif_audio = song[start:end]
    #    motif_audio.export('audio/snippets/queen-vanilla-ice/' + ds_name +
    #                       # "_Channels_" + str(len(df.index)) +
    #                       "_Motif_" + str(a) + '.wav', format="wav")
