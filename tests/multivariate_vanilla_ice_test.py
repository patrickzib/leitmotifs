import os

from audio.lyrics import *
from motiflets.motiflets import read_audio_from_dataframe

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

A_dataset = "Queen-David-Bowie-Under-Pressure"
A_ds_name = "Queen David Bowie - Under Pressure"

B_ds_name = "Vanilla Ice - Ice Ice Baby"
B_dataset = "Vanilla_Ice-Ice_Ice_Baby"

datasets = [(A_dataset, A_ds_name), (B_dataset, B_ds_name)]

k_max = 25
length_in_seconds = 3.4  # in seconds

# def test_read_write():
#     audio_file_url = path_to_wav + B_dataset + ".mp3"
#     pandas_file_url = path + B_dataset + ".csv"
#
#     audio_length_seconds, df, index_range = read_mp3(audio_file_url)
#     df.to_csv(pandas_file_url, compression='gzip')
#     audio_length_seconds2, df2, index_range2 = read_from_dataframe(pandas_file_url)


def test_audio():
    for dataset, ds_name in datasets:
        audio_file_url = path_to_wav + dataset + ".mp3"
        pandas_file_url = path + dataset + ".csv"
        audio_length_seconds, df, index_range = read_audio_from_dataframe(pandas_file_url)
        channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3']
        df = df.loc[channels]

        ml = Motiflets(
            ds_name, df,
            dimension_labels=df.index,
            n_dims=2
        )

        _, all_minima = ml.fit_motif_length(
            k_max, np.arange(50, 200, 10), subsample=2, plot=False)

        length_in_seconds = index_range[ml.motif_length]

        # best motiflet
        motiflet = np.sort(ml.motiflets[ml.elbow_points[-1]])
        print("Positions:", index_range[motiflet])

        # path_ = "audio/snippets/" + ds_name + \
        #        "_Channels_" + str(len(df.index)) + \
        #        "_Length_" + str(ml.motif_length) + \
        #        "_Motif.pdf"
        ml.plot_motifset()

        if os.path.isfile(audio_file_url):
            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets/",
                length_in_seconds, index_range, ml.motif_length, motiflet)
