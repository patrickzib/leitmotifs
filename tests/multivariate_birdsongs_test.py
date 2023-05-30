import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

from audio.lyrics import *
from pydub import AudioSegment

import audioread
import librosa

path = "../../motiflets_use_cases/birds/"

ks = 5
channels = 10
length_range = np.arange(25, 100, 1)
ds_name = "Common-Starling"
audio_file_url = path + "xc27154---common-starling---sturnus-vulgaris.mp3"

#ks = 10
#channels = 10
#length_range = np.arange(25, 50, 1)
#ds_name = "House-Sparrow"
#audio_file_url = path + "house-sparrow-passer-domesticus-audio.mp3"

def test_audio():
    # Read audio from wav file
    aro = audioread.ffdec.FFmpegAudioFile(audio_file_url)
    x, sr = librosa.load(aro, mono=True)
    mfcc_f = librosa.feature.mfcc(y=x, sr=sr)
    audio_length_seconds = librosa.get_duration(filename=audio_file_url)

    print("Length:", sr, "in seconds", audio_length_seconds, "s")

    index_range = np.arange(0, mfcc_f.shape[1]) * audio_length_seconds / mfcc_f.shape[1]
    df = pd.DataFrame(mfcc_f,
                      index=["MFCC " + str(a) for a in np.arange(0, mfcc_f.shape[0])],
                      columns = index_range)

    df.index.name = "MFCC"
    df = df.iloc[:channels]


    motif_length = plot_motif_length_selection(
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
        dimension_labels=df.index,
        motif_length=motif_length)

    #dists, motiflets, elbow_points, motif_length = ml.search_k_motiflets_elbow(
    #    ks,
    #    df.values,
    #    motif_length=motif_length,
    #    # elbow_deviation=1.25,
    #    slack=1.0,
    #)

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
        df,
        motifset=motiflet,
        dist=dists[elbow_points[0]],
        motif_length=motif_length,
        show=False)

    # plt.savefig(
    #    "audio/snippets/" + ds_name + "_Full_Channel_" + str(c) + "_Motif.pdf")
    plt.show()


    song = AudioSegment.from_mp3(audio_file_url)
    for a, motif in enumerate(motiflet):
        start = (index_range[motif]) * 1000  # ms
        end = start + length_in_seconds * 1000  # ms
        motif_audio = song[start:end]
        motif_audio.export('audio/bird_songs/' + ds_name +
                           "_Channel_" + str(channels) +
                           "_Motif_" + str(a) + '.wav', format="wav")