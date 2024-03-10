import os

import matplotlib as mpl

from motiflets.motiflets import read_audio_from_dataframe

mpl.rcParams['figure.dpi'] = 150

from audio.lyrics import *
from motiflets.competitors import *

# path outside the git
path_to_wav = "../../motiflets_use_cases/birds/"
path = "../datasets/audio/"
write_audio = False

datasets = {
    "Common-Starling": {
        "ks": [4],
        "motif_length": 50,
        "n_dims": 2,
        "slack": 1.0,
        "length_range": np.arange(25, 100, 5),
        "audio_file_url": path_to_wav + "xc27154---common-starling---sturnus-vulgaris.mp3",
        "pandas_file_url": path + "common-starling-sturnus-vulgaris.csv"
    },
    # "House-Sparrow": {
    #     "ks": [4],
    #     "n_dims": 10,
    #     "motif_length": 50,
    #     "slack": 1.0,
    #     "length_range": np.arange(25, 100, 5),
    #     "audio_file_url": path_to_wav + "house-sparrow-passer-domesticus-audio.mp3",
    #     "pandas_file_url": path + "house-sparrow-passer-domesticus.csv"
    # }
}

# dataset = datasets["Common-Starling"]
# dataset = datasets["House-Sparrow"]

channels = [
    'MFCC 0',
    'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
    'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
    'MFCC 11', 'MFCC 12', 'MFCC 13'
]


def get_ds_parameters(name):
    global ds_name, k_max, n_dims, length_range, motif_length
    global audio_file_url, pandas_file_url, ks, slack

    ds_name = name
    dataset = datasets[name]
    ks = dataset["ks"]
    k_max = np.max(ks) + 2
    n_dims = dataset["n_dims"]
    length_range = dataset["length_range"]
    slack = dataset["slack"]
    audio_file_url = dataset["audio_file_url"]
    pandas_file_url = dataset["pandas_file_url"]
    motif_length = dataset["motif_length"]


# def test_read_write():
#    audio_length_seconds, df, index_range = read_from_wav(audio_file_url)
#    df.to_csv(pandas_file_url, compression='gzip')
#    audio_length_seconds2, df2, index_range2 = read_from_dataframe()


def test_ground_truth():
    get_ds_parameters("Common-Starling")
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   ground_truth=ground_truth
                   )

    # print("Positions:", index_range[ground_truth.loc[0][0]])

    positions = []
    pos = np.array([[509, 559], [566, 616], [2229, 2279], [2137, 2187]])
    pos[:, 0] -= 15
    positions.append(pos)
    print(repr(positions))

    if os.path.isfile(audio_file_url):
        # extract motif sets
        for a, motif in enumerate(positions):
            motif_length = motif[a][1] - motif[a][0]
            length_in_seconds = motif_length * audio_length_seconds / df.shape[1]

            extract_audio_segment(
                df, ds_name, audio_file_url, "snippets",
                length_in_seconds, index_range, motif_length,
                np.array(motif)[:, 0], id=(a + 1))

    ml.plot_dataset()


def test_lama(
        dataset_name="Common-Starling",
        minimize_pairwise_dist=False,
        use_PCA=False,
        motifset_name="LAMA",
        plot=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        df_transform = pca.fit_transform(df.T).T
    else:
        df_transform = df

    ml = Motiflets(ds_name, df_transform,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   slack=slack,
                   minimize_pairwise_dist=minimize_pairwise_dist,
                   ground_truth=ground_truth,
                   )

    # learn parameters
    # motif_length, all_minima = ml.fit_motif_length(
    #     k_max, length_range,
    #     plot_motifsets=False
    # )

    # print("Best length", motif_length, length_in_seconds, "s")

    dists, motif_sets, elbow_points = ml.fit_k_elbow(
        k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifsets=False)

    print("Positions (Frame):", repr(np.sort(motif_sets[ks])))
    # print("Time:", repr(np.sort(index_range[np.int32(motif_sets[ks])])))

    if plot:
        ml.plot_motifset(motifset_name=motifset_name)

    if use_PCA:
        dims = [np.argsort(pca.components_[:])[:, :n_dims][0] for _ in ks]
    else:
        dims = ml.motiflets_dims[ks]

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:")
        print("\tpos\t:", repr(motiflet))
        print("\tdims\t:", repr(dims))

    # if write_audio:
    length_in_seconds = index_range[motif_length]
    if os.path.isfile(audio_file_url):
        extract_audio_segment(
            df, ds_name, audio_file_url, "bird_songs",
            length_in_seconds, index_range, motif_length,
            ml.motiflets[ml.elbow_points[-1]])

    return motif_sets[ks], dims


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)

    offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[offset[i]:, 0], data[offset[i]:, 1]

            ax[i].specgram(left,
                           Fs=samplingFrequency,
                           cmap='Grays',
                           # scale='dB',
                           vmin=-30, vmax=30
                           )
            ax[i].set_ylabel("Freq.")

            ax[i].set_ylim([0, 10000])
            ax[i].set_xlim([0, 0.92])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig("images_paper/bird_songs/spectrogram.pdf")


def test_plot_spectrogram():
    audio_file_urls = \
        ["images_paper/bird_songs/Common-Starling_Dims_10_Length_50_Motif_0.wav",
         "images_paper/bird_songs/Common-Starling_Dims_10_Length_50_Motif_1.wav",
         "images_paper/bird_songs/Common-Starling_Dims_10_Length_50_Motif_2.wav",
         "images_paper/bird_songs/Common-Starling_Dims_10_Length_50_Motif_3.wav"]

    plot_spectrogram(audio_file_urls)
    plt.show()


def test_emd_pca(dataset_name="Common-Starling", plot=True):
    return test_lama(dataset_name, use_PCA=True, motifset_name="PCA", plot=plot)


def test_mstamp(dataset_name="Common-Starling", plot=True, use_mdl=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)
    return run_mstamp(df, ds_name, motif_length=motif_length,
                      ground_truth=ground_truth, plot=plot,
                      use_mdl=use_mdl, use_dims=n_dims)


def test_kmotifs(dataset_name="Common-Starling", first_dims=True, plot=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels[:n_dims])

    motif_sets = []
    used_dims = []
    for target_k in ks:
        motif, dims = run_kmotifs(
            df,
            ds_name,
            motif_length=motif_length,
            slack=slack,
            r_ranges=np.arange(1, 100, 1),
            use_dims=n_dims if first_dims else df.shape[0],  # first dims or all dims
            target_k=target_k,
            ground_truth=ground_truth,
            plot=plot
        )
        used_dims.append(np.arange(dims))
        motif_sets.append(motif)

    return motif_sets, used_dims


def test_publication():
    dataset_names = [
        "Common-Starling"
    ]

    plot = False
    for dataset_name in dataset_names:
        motifA, dimsA = test_lama(dataset_name, plot=plot)
        motifG, dimsG = test_lama(dataset_name, plot=plot, minimize_pairwise_dist=True)
        motifB, dimsB = test_emd_pca(dataset_name, plot=plot)
        motifC, dimsC = test_mstamp(dataset_name, plot=plot, use_mdl=True)
        motifF, dimsF = test_mstamp(dataset_name, plot=plot, use_mdl=False)
        motifD, dimsD = test_kmotifs(dataset_name, first_dims=True, plot=plot)
        motifE, dimsE = test_kmotifs(dataset_name, first_dims=False, plot=plot)

        df = pd.DataFrame(columns=[
            "dataset", "k",
            "LAMA", "EMD", "mSTAMP+MDL", "mSTAMP",
            "K-Motifs (1st dims)", "K-Motifs (all dims)",
            "LAMA (naive)",
            "LAMA_dims", "EMD_dims", "mSTAMP_MDL_dims", "mSTAMP_dims",
            "K-Motifs (1st dims)_dims",
            "K-Motifs (all dims)_dims",
            "LAMA (naive)_dims"])

        for i, k in enumerate(ks):
            df.loc[len(df.index)] \
                = [dataset_name, k,
                   motifA[i].tolist(), motifB[i].tolist(), motifC[0], motifF[0],
                   motifD[i].tolist(), motifE[i].tolist(), motifG[i].tolist(),
                   dimsA[i].tolist(), dimsB[i].tolist(), dimsC[0].tolist(),
                   dimsF[0].tolist(),
                   dimsD[i].tolist(), dimsE[i].tolist(), dimsG[i].tolist()]

        print("--------------------------")
        print("LAMA:        \t", motifA, dimsA)
        print("EMD*:        \t", motifB, dimsB)
        print("mSTAMP+MDL:  \t", motifC, dimsC)
        print("mSTAMP:      \t", motifF, dimsF)
        print("K-Motifs (1st dims):\t", motifD, dimsD)
        print("K-Motifs (all dims):\t", motifE, dimsE)
        print("LAMA (naive): \t", motifG, dimsG)

        # from datetime import datetime
        # currentDateTime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        df.to_parquet(
            f'results/results_birdsounds_{dataset_name}.gzip',  # _{currentDateTime}
            compression='gzip')


def test_plot_results():
    dataset_names = [
        "Common-Starling"
    ]

    results = []

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        audio_length_seconds, df, index_range, ground_truth \
            = read_audio_from_dataframe(pandas_file_url, channels)

        df_loc = pd.read_parquet(
            f"results/results_birdsounds_{dataset_name}.gzip")

        id = df_loc.shape[0] - 1  # last index
        motifs = [
            # mSTAMP+MDL
            df_loc.loc[id]["mSTAMP+MDL"],
            # mSTAMP using top f dims
            df_loc.loc[id]["mSTAMP"],
            # LAMA
            df_loc.loc[id]["LAMA"],
            # LAMA + with naive
            df_loc.loc[id]["LAMA (naive)"],
            # EMD*
            df_loc.loc[id]["EMD"],
            # K-Motif
            df_loc.loc[id]["K-Motifs (1st dims)"],
            df_loc.loc[id]["K-Motifs (all dims)"],
        ]

        dims = [
            # mSTAMP+MDL
            df_loc.loc[id]["mSTAMP_MDL_dims"],
            # mSTAMP using top f dims
            df_loc.loc[id]["mSTAMP_dims"],
            # LAMA
            df_loc.loc[id]["LAMA_dims"],
            # LAMA + with naive
            df_loc.loc[id]["LAMA (naive)_dims"],
            # EMD*
            df_loc.loc[id]["EMD_dims"],
            # K-Motif
            df_loc.loc[id]["K-Motifs (1st dims)_dims"],
            df_loc.loc[id]["K-Motifs (all dims)_dims"],
        ]

        for method, motif_set in zip(
                ["mSTAMP+MDL", "mSTAMP", "LAMA", "LAMA (naive)", "EMD*", "K-Motifs (TOP-f)",
                 "K-Motifs (all)"], motifs):
            precision, recall = compute_precision_recall(
                np.sort(motif_set), ground_truth.values[0, 0], motif_length)
            results.append([ds_name, method, precision, recall])

        print(results)

        if False:
            motifset_names = [
                "mSTAMP+MDL",
                "mSTAMP",
                "LAMA",
                "LAMA (naive)",
                "EMD*",
                "K-Motifs (TOP-f)",
                "K-Motifs (all)"]

            out_path = "results/images/" + dataset_name + "_new.pdf"

            plot_motifsets(
                ds_name,
                df,
                motifsets=motifs,
                motiflet_dims=dims,
                motifset_names=motifset_names,
                motif_length=motif_length,
                ground_truth=ground_truth,
                show=out_path is None)

            if out_path is not None:
                plt.savefig(out_path)
                plt.show()

    pd.DataFrame(
        data=np.array(results),
        columns=["Dataset", "Method", "Precision", "Recall"]).to_csv(
        "results/birdsounds_precision.csv")
