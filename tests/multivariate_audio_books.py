import os
import matplotlib as mpl
from motiflets.motiflets import read_audio_from_dataframe

mpl.rcParams['figure.dpi'] = 150

from audio.lyrics import *

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

dataset = "The Theory of Everything First Lecture.wav"
dataset_df = "The-Theory-of-Everything-First-Lecture.csv"

k_max = 10
n_dims = 3
motif_length_range_in_s = np.arange(2.0, 2.5, 0.1)
ds_name = "The Theory of Everything First Lecture"
audio_file_url = path_to_wav + dataset
pandas_file_url = path + dataset_df


# def test_read_write():
#    audio_length_seconds, df, index_range = read_from_wav(audio_file_url)
#    df.to_csv(pandas_file_url, compression='gzip')
#    audio_length_seconds2, df2, index_range2 = read_from_dataframe(pandas_file_url)


def test_audio():
    audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3']
    df = df.loc[channels].iloc[:, :20000]

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   )
    ml.plot_dataset()

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

    if os.path.isfile(audio_file_url):
        # best motiflets
        for a, eb in enumerate(ml.elbow_points):
            motiflet = np.sort(ml.motiflets[eb])
            print("Positions:", motiflet, index_range[motiflet])

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))
