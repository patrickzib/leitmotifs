import matplotlib as mpl
import os

from audio.lyrics import *
from leitmotifs.competitors import *
from leitmotifs.lama import read_audio_from_dataframe

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path_to_wav = "../../motiflets_use_cases/audio/"
path = "../datasets/audio/"
write_audio = False


datasets = {
    "Numb - Linkin Park": {
        "ks": [5],
        "n_dims": 5,
        "motif_length": 226,
        "length_range_in_seconds": np.arange(5.0, 6.0, 0.25),
        "slack": 1.0,
        "ds_name": "Numb - Linkin Park",
        "audio_file_url": path_to_wav + "Numb - Linkin Park.wav",
        "pandas_file_url": path + "Numb-Linkin-Park.csv",
        "lrc_url": path_to_wav + "Numb - Linkin Park.lrc",
    },
    "What I've Done - Linkin Park": {
        "ks": [6],
        "n_dims": 3,
        "motif_length": 300,
        "length_range_in_seconds": np.arange(19.0, 21, 0.25),
        "slack": 1.0,
        "ds_name": "What I've Done - Linkin Park",
        "audio_file_url": path_to_wav + "What I've Done - Linkin Park.wav",
        "pandas_file_url": path + "What-I-ve-Done-Linkin-Park.csv",
        "lrc_url": path_to_wav + "What I've Done - Linkin Park.lrc"
    },
    "The Rolling Stones - Paint It, Black": {
        "ks": [10, 14],
        "n_dims": 3,
        "motif_length": 232,
        "length_range_in_seconds": np.arange(5.0, 6.0, 0.1),
        "slack": 0.5,
        "ds_name": "The Rolling Stones - Paint It, Black",
        "audio_file_url": path_to_wav + "The Rolling Stones - Paint It, Black.wav",
        "pandas_file_url": path + "The-Rolling-Stones-Paint-It-Black.csv",
        "lrc_url": path_to_wav + "The Rolling Stones - Paint It, Black.lrc"
    },
    "Vanilla Ice - Ice Ice Baby": {
        "ks": [20],
        "n_dims": 2,
        "motif_length": 170,
        "length_range_in_seconds": np.arange(160, 190, 10),
        "slack": 0.5,
        "ds_name": "Vanilla Ice - Ice Ice Baby",
        "audio_file_url": path_to_wav + "Vanilla_Ice-Ice_Ice_Baby.mp3",
        "pandas_file_url": path + "Vanilla_Ice-Ice_Ice_Baby.csv",
        "lrc_url": None
    },
    "Queen David Bowie - Under Pressure": {
        "ks": [16],
        "n_dims": 3,
        "motif_length": 180,
        "length_range_in_seconds": np.arange(170, 190, 10),
        "slack": 0.5,
        "ds_name": "Queen David Bowie - Under Pressure",
        "audio_file_url": path_to_wav + "Queen-David-Bowie-Under-Pressure.mp3",
        "pandas_file_url": path + "Queen-David-Bowie-Under-Pressure.csv",
        "lrc_url": None
    }
}

# dataset = datasets["The Rolling Stones - Paint It, Black"]
# dataset = datasets["What I've Done - Linkin Park"]
# dataset = datasets["Numb - Linkin Park"]
# dataset = datasets["Vanilla Ice - Ice Ice Baby"]
# dataset = datasets["Queen David Bowie - Under Pressure"]


def get_ds_parameters(name):
    global ks, n_dims, slack, ds_name, audio_file_url, pandas_file_url
    global lrc_url, motif_length, k_max, motif_length_range_in_s

    dataset = datasets[name]
    ks = dataset["ks"]
    n_dims = dataset["n_dims"]
    slack = dataset["slack"]
    ds_name = dataset["ds_name"]
    audio_file_url = dataset["audio_file_url"]
    pandas_file_url = dataset["pandas_file_url"]
    lrc_url = dataset["lrc_url"]
    motif_length = dataset["motif_length"]

    # for learning parameters
    k_max = np.max(dataset["ks"]) + 2
    motif_length_range_in_s = dataset["length_range_in_seconds"]


channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4',
            'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
            ]



def test_learn_parameters():
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    n_dims = 3
    ml = LAMA(ds_name, df,
              dimension_labels=df.index,
              n_dims=n_dims,
              ground_truth=ground_truth,
              slack=slack
              )

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])
    m, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=False,
        plot_motifsets=True,
        plot_best_only=False
    )

    ml.plot_motifset(motifset_name="LAMA")

    length_in_seconds = m * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, m)

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.leitmotifs[eb])
        print("Positions:", index_range[motiflet])
        print("Positions:", list(zip(motiflet, motiflet + motif_length)))

        if write_audio:
            if os.path.isfile(audio_file_url):
                # extract motif sets
                extract_audio_segment(
                    df, ds_name, audio_file_url, "snippets",
                    length_in_seconds, index_range, m, motiflet, id=(a + 1))


def test_publication():
    test_lama()
    test_emd_pca()
    test_mstamp()
    test_kmotifs()


def test_lama(
        dataset_name="Vanilla Ice - Ice Ice Baby",
        minimize_pairwise_dist=False,
        use_PCA=False,
        motifset_name="LAMA",
        plot=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels=channels)

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        df_transform = pca.fit_transform(df.T).T
    else:
        df_transform = df

    ml = LAMA(
        ds_name, df_transform,
        dimension_labels=df.index,
        n_dims=n_dims,
        ground_truth=ground_truth,
        minimize_pairwise_dist=minimize_pairwise_dist,
        slack=slack
    )

    # motif_length_range = np.int32(motif_length_range_in_s /
    #                               audio_length_seconds * df.shape[1])
    # m, _ = ml.fit_motif_length(
    #     k_max,
    #     motif_length_range,
    #     plot=False,
    #     plot_elbows=True,
    #     plot_motifsets=True,
    #     plot_best_only=True
    # )

    dists, motif_sets, elbow_points = ml.fit_k_elbow(
        k_max=k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifsets=False,
    )

    if plot:
        ml.elbow_points = ks
        ml.plot_motifset(motifset_name=motifset_name)

    if use_PCA:
        dims = [np.argsort(pca.components_[:])[:, :n_dims][0] for _ in ks]
    else:
        dims = ml.leitmotifs_dims[ks]

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    print("Positions:")
    # for a, eb in enumerate(ml.elbow_points):
    for a, eb in enumerate(ks):
        motiflet = np.sort(ml.leitmotifs[eb])
        print("\tMotif pos\t:", repr(motiflet))
        print("\tdims\t:", repr(dims))

        if write_audio:
            if os.path.isfile(audio_file_url):
                # extract motif sets
                extract_audio_segment(
                    df, ds_name, audio_file_url, "snippets",
                    length_in_seconds, index_range, motif_length, motiflet, id=(a + 1))

    return motif_sets[ks], dims


def test_emd_pca(dataset_name="The Rolling Stones - Paint It, Black", plot=True):
    return test_lama(dataset_name, use_PCA=True, motifset_name="PCA", plot=plot)


def test_mstamp(dataset_name="The Rolling Stones - Paint It, Black",
                plot=True, use_mdl=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    return run_mstamp(df, ds_name, motif_length=motif_length,
                      ground_truth=ground_truth, plot=plot,
                      use_mdl=use_mdl, use_dims=n_dims)


def test_kmotifs(dataset_name="The Rolling Stones - Paint It, Black",
                 first_dims=True, plot=True):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url)

    motif_sets = []
    used_dims = []
    for target_k in ks:
        motif, dims = run_kmotifs(
            df,
            ds_name,
            motif_length=motif_length,
            slack=slack,
            r_ranges=np.arange(10, 600, 5),
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
        "The Rolling Stones - Paint It, Black",
        "What I've Done - Linkin Park",
        "Numb - Linkin Park",
        "Vanilla Ice - Ice Ice Baby",
        "Queen David Bowie - Under Pressure"
    ]
    method_names = [
        "LAMA",
        "LAMA (naive)",
        "mSTAMP+MDL",
        "mSTAMP",
        "EMD*",
        "K-Motifs (TOP-f)",
        "K-Motifs (all)"
    ]
    plot = False
    for dataset_name in dataset_names:
        motifA, dimsA = test_lama(dataset_name, plot=plot)
        motifB, dimsB = test_lama(dataset_name, plot=plot, minimize_pairwise_dist=True)
        motifC, dimsC = test_mstamp(dataset_name, plot=plot, use_mdl=True)
        motifD, dimsD = test_mstamp(dataset_name, plot=plot, use_mdl=False)
        motifE, dimsE = test_emd_pca(dataset_name, plot=plot)
        motifF, dimsF = test_kmotifs(dataset_name, first_dims=True, plot=plot)
        motifG, dimsG = test_kmotifs(dataset_name, first_dims=False, plot=plot)

        method_names_dims = [name+"_dims" for name in method_names]
        columns = ["dataset", "k"]
        columns.extend(method_names)
        columns.extend(method_names_dims)
        df = pd.DataFrame(columns=columns)

        for i, k in enumerate(ks):
            df.loc[len(df.index)] \
                = [dataset_name, k,
                   motifA[i].tolist(), motifB[i].tolist(), motifC[0], motifD[0],
                   motifE[i].tolist(), motifF[i].tolist(), motifG[i].tolist(),
                   dimsA[i].tolist(), dimsB[i].tolist(), dimsC[0].tolist(), dimsD[0].tolist(),
                   dimsE[i].tolist(), dimsF[i].tolist(), dimsG[i].tolist()]

        # from datetime import datetime
        # currentDateTime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        df.to_parquet(
            f'results/results_audio_{dataset_name}.gzip',  # _{currentDateTime}
            compression='gzip')


def test_plot_results():
    dataset_names = [
        "The Rolling Stones - Paint It, Black",
        "What I've Done - Linkin Park",
        "Numb - Linkin Park",
        "Vanilla Ice - Ice Ice Baby",
        "Queen David Bowie - Under Pressure"
    ]
    method_names = [
        "LAMA",
        "LAMA (naive)",
        "mSTAMP+MDL",
        "mSTAMP",
        "EMD*",
        "K-Motifs (TOP-f)",
        "K-Motifs (all)"
    ]
    results = []

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        audio_length_seconds, df, index_range, ground_truth \
            = read_audio_from_dataframe(pandas_file_url, channels)

        df_loc = pd.read_parquet(
            f"results/results_audio_{dataset_name}.gzip")

        motifs = []
        dims = []
        for id in range(df_loc.shape[0]):
            for motif_method in method_names:
                motifs.append(df_loc.loc[id][motif_method])
                dims.append(df_loc.loc[id][motif_method + "_dims"])

        # write results to file
        for id in range(df_loc.shape[0]):
            for method, motif_set in zip(
                    method_names,
                    motifs[id * len(method_names): (id + 1) * len(method_names)]
            ):
                precision, recall = compute_precision_recall(
                    np.sort(motif_set), ground_truth.values[0, 0], motif_length)
                results.append([ds_name, method, precision, recall])

        #pd.DataFrame(
        #    data=np.array(results),
        #    columns=["Dataset", "Method", "Precision", "Recall"]).to_csv(
        #    "results/audio_precision.csv")

        print(results)

        if True:
            plot_names = [
                "mSTAMP+MDL",
                "mSTAMP",
                "EMD*",
                "K-Motifs (all)",
                "LAMA",
            ]

            motifset_names = []
            positions = []

            for a, id in enumerate(range(df_loc.shape[0])):  # last index
                motifset_names.extend(method_names)
                pos = np.array([method_names.index(name) for name in plot_names])
                positions.extend(pos + len(method_names) * a)

            out_path = "results/images/" + dataset_name + "_new.pdf"
            plot_motifsets(
                ds_name,
                df,
                motifsets=[motifs[pos] for pos in positions],
                leitmotif_dims=[dims[pos] for pos in positions],
                motifset_names=plot_names,
                motif_length=motif_length,
                ground_truth=ground_truth,
                show=out_path is None)

            if out_path is not None:
                plt.savefig(out_path)
                plt.show()


def plot_spectrogram(audio_file_urls):
    fig, ax = plt.subplots(len(audio_file_urls), 1,
                           figsize=(10, 5),
                           sharex=True, sharey=True)

    # offset = [3000, 3000, 10000, 10000]
    for i, audio_file_url in enumerate(audio_file_urls):
        if os.path.isfile(audio_file_url):
            samplingFrequency, data = read_wave(audio_file_url)
            left, right = data[:, 0], data[:, 1]

            ax[i].specgram(left,
                           Fs=samplingFrequency,  # cmap='Grays',
                           # scale='dB', # vmin=-50, vmax=0
                           )
            ax[i].set_ylabel("Freq.")

            ax[i].set_ylim([0, 5000])
            # ax[i].set_xlim([0, 0.92])
        else:
            raise ("No audio file found.")

    ax[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig("images_paper/audio/rolling-stones-spectrogram_new.pdf")


def test_plot_spectrogram():
    audio_file_urls = \
        [
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_0.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_1.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_2.wav",
            "images_paper/audio/The Rolling Stones - Paint It, Black_Dims_7_Length_232_Motif_2_3.wav"]

    plot_spectrogram(audio_file_urls)
    plt.show()