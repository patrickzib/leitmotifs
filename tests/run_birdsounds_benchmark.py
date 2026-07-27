import os

import matplotlib as mpl

from _repo_path import DATASETS_DIR, EXTERNAL_USE_CASES_DIR, TESTS_DIR
from leitmotifs.lama import read_audio_from_dataframe

mpl.rcParams['figure.dpi'] = 150

from audio.lyrics import *
from leitmotifs.competitors import *

# path outside the git
path_to_wav = str(EXTERNAL_USE_CASES_DIR / "birds") + "/"
path = str(DATASETS_DIR / "audio") + "/"
results_path = TESTS_DIR / "results"
WRITE_AUDIO = False

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

    ml = LAMA(ds_name, df,
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

    if WRITE_AUDIO and os.path.isfile(audio_file_url):
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
        distance="znormed_ed",
        exclusion_range=None,
        plot=True,
        write_audio=WRITE_AUDIO):
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

    ml = LAMA(
        ds_name, df_transform,
        dimension_labels=df.index,
        distance=distance,
        n_dims=n_dims,
        minimize_pairwise_dist=minimize_pairwise_dist,
        ground_truth=ground_truth,
        slack=exclusion_range if exclusion_range else slack
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

    if plot:
        ml.plot_motifset(motifset_name=motifset_name)

    if use_PCA:
        dims = [np.argsort(pca.components_[:])[:, :n_dims][0] for _ in ks]
    else:
        dims = ml.leitmotifs_dims[ks]

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.leitmotifs[eb])
        print(f"Motif {a + 1} positions [s]: {format_seconds(index_range[motiflet])}")
        print(f"Motif {a + 1} start positions [frames]: {as_int_list(motiflet)}")
        print(f"Motif {a + 1} dims: {format_motif_dims(dims, a)}")

    if write_audio:
        length_in_seconds = index_range[motif_length]
        if os.path.isfile(audio_file_url):
            extract_audio_segment(
                df, ds_name, audio_file_url, "bird_songs",
                length_in_seconds, index_range, motif_length,
                ml.leitmotifs[ml.elbow_points[-1]])

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


def test_emd_pca(
        dataset_name="Common-Starling",
        plot=True,
        write_audio=WRITE_AUDIO):
    return test_lama(
        dataset_name,
        use_PCA=True,
        motifset_name="PCA",
        plot=plot,
        write_audio=write_audio)


def test_mstamp(
        dataset_name="Common-Starling",
        plot=True,
        use_mdl=True,
        write_audio=WRITE_AUDIO):
    get_ds_parameters(dataset_name)
    audio_length_seconds, df, index_range, ground_truth \
        = read_audio_from_dataframe(pandas_file_url, channels)
    return run_mstamp(df, ds_name, motif_length=motif_length,
                      ground_truth=ground_truth, plot=plot,
                      use_mdl=use_mdl, use_dims=n_dims)


def test_kmotifs(
        dataset_name="Common-Starling",
        first_dims=True,
        plot=True,
        write_audio=WRITE_AUDIO):
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


def test_publication(plot=False, method_names=None, write_audio=WRITE_AUDIO):
    dataset_names = [
        "Common-Starling"
    ]
    if method_names is None:
        method_names = [
            "LAMA",
            "LAMA (naive)",
            "mSTAMP+MDL",
            "mSTAMP",
            "EMD*",
            "K-Motifs (TOP-f)",
            "K-Motifs (all)",
            "LAMA (cid)",
            "LAMA (ed)",
            "LAMA (cosine)"
        ]

    file_prefix = "results_birdsounds"

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        run_tests(
            dataset_name,
            ks=ks,
            method_names=method_names,
            test_lama=test_lama,
            test_mstamp=test_mstamp,
            test_emd_pca=test_emd_pca,
            test_kmotifs=test_kmotifs,
            file_prefix=file_prefix,
            plot=plot,
            write_audio=write_audio,
        )


def test_plot_results(
        plot=True, method_names=None, all_plot_names=None):
    dataset_names = [
        "Common-Starling"
    ]
    if method_names is None:
        method_names = [
            "LAMA",
            "LAMA (naive)",
            "mSTAMP+MDL",
            "mSTAMP",
            "EMD*",
            "K-Motifs (TOP-f)",
            "K-Motifs (all)",
            "LAMA (cid)",
            "LAMA (ed)",
            "LAMA (cosine)"
        ]

    results = []
    if all_plot_names is None:
        all_plot_names = {
            "_new": [
                "mSTAMP+MDL",
                "mSTAMP",
                "EMD*",
                "K-Motifs (all)",
                "LAMA",
            ], "_distances": [
                "LAMA",
                "LAMA (cid)",
                "LAMA (ed)",
                "LAMA (cosine)"
            ]
        }

    file_prefix = "results_birdsounds"
    output_file = "birdsounds_precision"

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        audio_length_seconds, df, index_range, ground_truth \
            = read_audio_from_dataframe(pandas_file_url, channels)

        eval_tests(
            dataset_name,
            ds_name,
            df,
            method_names,
            motif_length,
            ground_truth,
            all_plot_names,
            file_prefix,
            results,
            plot=plot
        )

    results_path.mkdir(parents=True, exist_ok=True)
    benchmark_results_dataframe(results).to_csv(results_path / f"{output_file}.csv")
    print_benchmark_summary(results)


def main():
    test_publication(write_audio=False)
    test_plot_results(plot=False)


if __name__ == "__main__":
    main()
