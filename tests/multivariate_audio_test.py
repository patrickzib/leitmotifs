import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import matplotlib
from audio.lyrics import *

import wave
import librosa
from os.path import exists

ks = 10
ds_name = "Numb - Linkin Park"
path = "../../motiflets_use_cases/audio/"
audio_file_url= path + "Numb - Linkin Park.wav"
lrc_url= path + "Numb - Linkin Park.lrc"


def test_audio():
    # Read audio from wav file
    x, sr, mfcc_f, audio_length_seconds = read_audio(audio_file_url, True)

    print("Length:", audio_length_seconds)
    channels = 3

    index_range = np.arange(0, mfcc_f.shape[1]) * audio_length_seconds / mfcc_f.shape[1]
    df = pd.DataFrame(mfcc_f,
                      index=["MFCC_" + str(a) for a in np.arange(0, 20)],
                      #columns=index_range
                      )
    df.index.name="MFCC"
    df = df.iloc[:channels,:]

    length_in_seconds = 4
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
        l = (lookup_lyrics(df_sub, index_range[m], length_in_seconds))
        lyrics.append(l)
        print(i+1, l)

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