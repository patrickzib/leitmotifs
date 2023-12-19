import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

path = "../../motiflets_use_cases/audio/"

A_dataset = "Queen-David-Bowie-Under-Pressure"
A_ds_name = "Queen David Bowie - Under Pressure"

B_ds_name = "Vanilla Ice - Ice Ice Baby"
B_dataset = "Vanilla_Ice-Ice_Ice_Baby"

datasets = [(A_dataset, A_ds_name), (B_dataset, B_ds_name)]

k_max = 25
length_in_seconds = 3.4  # in seconds


def test_audio():
    for dataset, ds_name in datasets:
        audio_file_url = path + dataset + ".mp3"
        audio_length_seconds, df, index_range = read_mp3(audio_file_url)
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

        path_ = "audio/snippets/" + ds_name + \
                "_Channels_" + str(len(df.index)) + \
                "_Length_" + str(ml.motif_length) + \
                "_Motif.pdf"
        ml.plot_motifset(path=path_)

        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets/",
            length_in_seconds, index_range, ml.motif_length, motiflet)
