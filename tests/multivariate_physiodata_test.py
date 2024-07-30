import matplotlib as mpl
import pandas as pd

mpl.rcParams['figure.dpi'] = 150

from leitmotifs.competitors import *
from leitmotifs.lama import *

# Experiment with different noise levels to show robustness of the method
noise_level = None

def znormalize(ts):
    for i in range(3):
        ts[:, 3*i : 3*(i+1)] \
            = ((ts[:, 3*i : 3*(i+1)] - np.mean(ts[:, 3*i : 3*(i+1)], axis=None)) /
               np.std(ts[:, 3*i : 3*(i+1)], axis=None))
    return ts


def load_physiodata():
    global noise_level

    # subjects = range(1, 6)
    # exercises = range(1, 9)
    # relevant_imus = np.array([2, 4, 2, 2, 2, 2, 2, 2])
    #
    # root_path = "../datasets/physiodata"
    # df = pd.DataFrame(columns=['subject', 'exercise', 'imu', 'ts'])
    #
    # for subject in subjects:
    #     for exercise, imu in zip(exercises, relevant_imus):
    #         path = os.path.join(root_path, f"s{subject}", f"e{exercise}", f"u{imu}",
    #                             "test.txt")
    #         f = open(path)
    #         next(f)
    #         data = np.array([l.split(';') for l in f.readlines()], dtype=np.float64)
    #         # acc (3 spatial axes), gyr (3 spatial axes), mag (3 spatial axes)
    #         df.loc[len(df.index)] = [subject, exercise, imu, data[:, 1:]]
    #         f.close()
    # df['ts'] = df['ts'].apply(znormalize)
    # subject = 2
    # exercise = 1
    # imu = 2
    # print(f"Subject {subject}, Exercise {exercise}, IMU {imu}")
    # *_, ts = df.query('subject == @subject & exercise == @exercise & imu == @imu').iloc[0]
    # df2 = pd.DataFrame(ts.T)
    # # df2.to_csv("../datasets/physiodata/physio.csv", index=False)


    df_gt = read_ground_truth("../datasets/physiodata/physio")
    df = pd.read_csv("../datasets/physiodata/physio.csv")

    if noise_level:
        print ("Adding noise to the data", noise_level)
        df = add_gaussian_noise(df, noise_level)

    return df, df_gt


datasets = {
    "Physiodata": {
        "ks": [20],
        "motif_length": 150,
        "n_dims": 3,
        "slack": 0.5,
        "length_range": np.arange(150, 180, 10),
    },
}


def get_ds_parameters(name="Physiodata"):
    global ds_name, k_max, n_dims, length_range, motif_length
    global audio_file_url, pandas_file_url, ks, slack

    ds_name = name
    dataset = datasets[name]
    ks = dataset["ks"]
    k_max = np.max(ks) + 2
    n_dims = dataset["n_dims"]
    length_range = dataset["length_range"]
    slack = dataset["slack"]
    motif_length = dataset["motif_length"]


def test_lama(
        dataset_name="Physiodata",
        minimize_pairwise_dist=False,
        use_PCA=False,
        motifset_name="LAMA",
        distance="znormed_ed",
        plot=True):
    get_ds_parameters(dataset_name)
    df, ground_truth = load_physiodata()

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        df_transform = pca.fit_transform(df.T).T
    else:
        df_transform = df

    ml = LAMA(ds_name, df_transform,
              dimension_labels=df.index,
              distance=distance,
              n_dims=n_dims,
              slack=slack,
              minimize_pairwise_dist=minimize_pairwise_dist,
              ground_truth=ground_truth)

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

    if plot:
        ml.elbow_points = ks
        ml.plot_motifset(motifset_name=motifset_name)

    if use_PCA:
        dims = [np.argsort(pca.components_[:])[:, :n_dims][0] for _ in ks]
    else:
        dims = ml.leitmotifs_dims[ks]

    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.leitmotifs[eb])
        print("Positions:")
        print("\tpos\t:", repr(motiflet))
        print("\tdims\t:", repr(dims))

    return motif_sets[ks], dims


def test_emd_pca(dataset_name="Physiodata", plot=True):
    return test_lama(dataset_name, use_PCA=True, motifset_name="PCA", plot=plot)


def test_mstamp(dataset_name="Physiodata", plot=True, use_mdl=True):
    get_ds_parameters(dataset_name)
    df, ground_truth = load_physiodata()

    return run_mstamp(df, ds_name, motif_length=motif_length,
                      ground_truth=ground_truth, plot=plot,
                      use_mdl=use_mdl, use_dims=n_dims)


def test_kmotifs(dataset_name="Physiodata", first_dims=True, plot=True):
    get_ds_parameters(dataset_name)
    df, ground_truth = load_physiodata()

    motif_sets = []
    used_dims = []
    for target_k in ks:
        motif, dims = run_kmotifs(
            df,
            ds_name,
            motif_length=motif_length,
            slack=slack,
            r_ranges=np.arange(150, 300, 5),
            use_dims=n_dims if first_dims else df.shape[0],  # first dims or all dims
            target_k=target_k,
            ground_truth=ground_truth,
            plot=plot
        )
        used_dims.append(np.arange(dims))
        motif_sets.append(motif)

    return motif_sets, used_dims


def test_publication(plot=False):
    dataset_names = [
        "Physiodata"
    ]
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

    if noise_level:
        print ("Adding noise to the data", noise_level)
        file_prefix = "results_physio_"+str(noise_level)
    else:
        file_prefix = "results_physio"

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
            plot=plot
        )


def test_plot_results(plot=True):
    dataset_names = [
        "Physiodata"
    ]
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
    if noise_level:
        print ("Adding noise to the data", noise_level)
        file_prefix = "results_physio_"+str(noise_level)
        output_file = "physio_precision_"+str(noise_level)
    else:
        file_prefix = "results_physio"
        output_file = "physio_precision"

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        df, ground_truth = load_physiodata()

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

    pd.DataFrame(
        data=np.array(results),
        columns=["Dataset", "Method", "Precision", "Recall"]).to_csv(
        "results/"+output_file+".csv")