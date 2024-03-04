import os

from audio.lyrics import *
from motiflets.motiflets import read_audio_from_dataframe
from motiflets.motiflets import read_ground_truth
from motiflets.competitors import *

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"

A_dataset = "Queen-David-Bowie-Under-Pressure"
A_ds_name = "Queen David Bowie - Under Pressure"

B_ds_name = "Vanilla Ice - Ice Ice Baby"
B_dataset = "Vanilla_Ice-Ice_Ice_Baby"

datasets = [(A_dataset, A_ds_name), (B_dataset, B_ds_name)]

m = [180, 180]
k_max = [18, 22]
dims = [5, 2]
ks = [16, 20]
# length_in_seconds = 3.4  # in seconds

channels = [
    'MFCC 0',
    'MFCC 1', 'MFCC 2', 'MFCC 3',
    'MFCC 4', 'MFCC 5', 'MFCC 6',
    'MFCC 7', 'MFCC 8', 'MFCC 9',
    'MFCC 10',
    # 'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15',
]


def test_read_write():
    audio_file_url = path_to_wav + A_dataset + ".mp3"
    pandas_file_url = path + A_dataset + ".csv"

    audio_length_seconds, df, index_range = read_mp3(audio_file_url)

    # fixes Queen problem with file-size
    # audio_length_seconds = 216.06
    # index_range = np.arange(0, audio_length_seconds, audio_length_seconds / df.shape[1])
    # df.columns = index_range

    # df.to_csv(pandas_file_url, compression='gzip')
    # audio_length_seconds2, df2, index_range2 = read_from_dataframe(pandas_file_url)
    print("lengths", audio_length_seconds, len(df))


def test_audio():
    i = 0
    for dataset, ds_name in datasets:
        pandas_file_url = path + dataset + ".csv"
        audio_length_seconds, df, index_range = read_audio_from_dataframe(
            pandas_file_url)
        ground_truth = read_ground_truth(pandas_file_url, path="")
        df = df.loc[channels]

        ml = Motiflets(
            ds_name, df,
            dimension_labels=df.index,
            slack=0.6,
            n_dims=dims[i],
            ground_truth=ground_truth
        )

        # m, all_minima = ml.fit_motif_length(
        #     k_max[0],
        #     np.arange(170, 190, 10),
        #     plot=True,
        #     plot_elbows=True,
        #     plot_motifsets=False,
        #     plot_best_only=True,
        # )

        ml.fit_k_elbow(
            k_max[i],
            motif_length=m[i],
            plot_elbows=False,
            plot_motifsets=False,
        )

        ml.elbow_points = [ml.elbow_points[-1]]
        ml.plot_motifset(motifset_name="LAMA")

        print("Best length:", m, index_range[m])
        motiflet = np.sort(ml.motiflets[ks[i]])
        print("Positions:", motiflet)

        if False:
            audio_file_url = path_to_wav + dataset + ".mp3"
            length_in_seconds = index_range[ml.motif_length]
            if os.path.isfile(audio_file_url):
                extract_audio_segment(
                    df, ds_name, audio_file_url, "snippets/",
                    length_in_seconds, index_range, ml.motif_length, motiflet)

        i += 1
