import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

path = "../../motiflets_use_cases/audio/"

import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

path = "../../motiflets_use_cases/audio/"
dataset = "The Theory of Everything First Lecture.wav"

k_max = 10
n_dims = 3
motif_length_range_in_s = np.arange(2.0, 2.5, 0.1)
ds_name = "The Theory of Everything First Lecture"
audio_file_url = path + dataset


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


def test_audio():
    audio_length_seconds, df, index_range = read_songs()
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3']
    df = df.loc[channels].iloc[:, :20000]


    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   )
    #ml.plot_dataset()

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

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

    # best motiflets
    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:", motiflet, index_range[motiflet])

        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets",
            length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))
