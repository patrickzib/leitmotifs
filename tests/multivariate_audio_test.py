from audio.lyrics import *
from pydub import AudioSegment
import scipy.cluster.hierarchy as sch

#import matplotlib
#matplotlib.use('macosx')

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/audio/"

# channels = 3
ks = 10
length_in_seconds = 4
ds_name = "Numb - Linkin Park"
audio_file_url= path + "Numb - Linkin Park.wav"
lrc_url = path + "Numb - Linkin Park.lrc"

# channels = 3
#ks = 10
#length_in_seconds = 7.75
#ds_name = "What I've Done - Linkin Park"
#audio_file_url = path + "What I've Done - Linkin Park.wav"
#lrc_url = path + "What I've Done - Linkin Park.lrc"

# channels = 3
#ks = 15
#length_in_seconds = 5.4
#ds_name = "The Rolling Stones - Paint It, Black"
#audio_file_url = path + "The Rolling Stones - Paint It, Black.wav"
#lrc_url = path + "The Rolling Stones - Paint It, Black.lrc"



def test_dendrogram():
    channels = 10
    audio_length_seconds, df, index_range = read_songs()
    df = df.iloc[0:channels]

    motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length)

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

    k = 3
    y_dimensions = sch.fcluster(Z, k, criterion='maxclust')
    mapping = list(zip(y_dimensions, df.index))

    joint_clusters = {}
    for i in range(1, k + 1):
        print("Cluster", i)
        joint_clusters[i] = [x[1] for x in mapping if x[0] == i]
        print(joint_clusters[i])
        print("----")

def test_audio():
    audio_length_seconds, df, index_range = read_songs()

    # df = df.iloc[:channels]
    # channels = ['MFCC 0', 'MFCC 1', 'MFCC 2']
    channels = ['MFCC 0', 'MFCC 2', 'MFCC 3', 'MFCC 5', 'MFCC 9']

    """
    ['MFCC 7', 'MFCC 8']
    ----
    Cluster
    2
    ['MFCC 1', 'MFCC 4', 'MFCC 6']
    ----
    Cluster
    3
    ['MFCC 0', 'MFCC 2', 'MFCC 3', 'MFCC 5', 'MFCC 9']
    """

    df = df.loc[channels]

    motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length)

    subtitles = read_lrc(lrc_url)
    df_sub = get_dataframe_from_subtitle_object(subtitles)
    df_sub.set_index("seconds", inplace=True)

    dists, motiflets, elbow_points, motif_length = ml.search_k_motiflets_elbow(
        ks,
        df.values,
        motif_length=motif_length,
        elbow_deviation=1.25
    )

    # best motiflet
    motiflet = np.sort(motiflets[elbow_points[-1]])
    print("Positions:", index_range[motiflet])

    lyrics = []
    for i, m in enumerate(motiflet):
        l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
        lyrics.append(l)
        print(i + 1, l)

    name = ds_name + " " + lyrics[-1] + " (" + str(len(motiflet)) + "x)"

    plot_motiflet(
        df,
        motiflet,
        motif_length,
        title=lyrics[0]
    )
    plt.tight_layout()
    # plt.savefig("audio/snippets/" + ds_name + "_Channels_" + str(len(df.index)) + "_Motif.pdf")
    plt.show()

    plot_motifset(
        name,
        df,
        motifset=motiflet,
        dist=dists[elbow_points[0]],
        motif_length=motif_length, show=False)

    # plt.savefig(
    #    "audio/snippets/" + ds_name + "_Full_Channel_" + str(c) + "_Motif.pdf")
    plt.show()

    song = AudioSegment.from_wav(audio_file_url)
    for a, motif in enumerate(motiflet):
        start = (index_range[motif]) * 1000  # ms
        end = start + length_in_seconds * 1000  # ms
        motif_audio = song[start:end]
        motif_audio.export('audio/snippets/' + ds_name +
                           "_Channels_" + str(len(df.index)) +
                           "_Motif_" + str(a) + '.wav', format="wav")


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