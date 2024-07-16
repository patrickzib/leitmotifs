import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

from leitmotifs.competitors import *
from leitmotifs.lama import *

def znormalize(ts):
    for i in range(3):
        ts[:, 3*i : 3*(i+1)] \
            = ((ts[:, 3*i : 3*(i+1)] - np.mean(ts[:, 3*i : 3*(i+1)], axis=None)) /
               np.std(ts[:, 3*i : 3*(i+1)], axis=None))
    return ts


def load_physiodata():
    subjects = range(1, 6)
    exercises = range(1, 9)
    relevant_imus = np.array([2, 4, 2, 2, 2, 2, 2, 2])

    root_path = "../datasets/physiodata"
    df = pd.DataFrame(columns=['subject', 'exercise', 'imu', 'ts'])

    for subject in subjects:
        for exercise, imu in zip(exercises, relevant_imus):
            path = os.path.join(root_path, f"s{subject}", f"e{exercise}", f"u{imu}",
                                "test.txt")
            f = open(path)
            next(f)
            data = np.array([l.split(';') for l in f.readlines()], dtype=np.float64)
            # acc (3 spatial axes), gyr (3 spatial axes), mag (3 spatial axes)
            df.loc[len(df.index)] = [subject, exercise, imu, data[:, 1:]]
            f.close()
    df['ts'] = df['ts'].apply(znormalize)
    df_gt = read_ground_truth("../datasets/physiodata/physio")

    subject = 2
    exercise = 1
    imu = 2

    print(f"Subject {subject}, Exercise {exercise}, IMU {imu}")
    *_, ts = df.query('subject == @subject & exercise == @exercise & imu == @imu').iloc[0]

    return pd.DataFrame(ts.T), df_gt


datasets = {
    "Physiodata": {
        "ks": [18],
        "motif_length": 206,
        "n_dims": 3,
        "slack": 0.5,
        "length_range": np.arange(201, 282, 5),
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


def test_publication():
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
        "K-Motifs (all)"
    ]
    plot = True
    for dataset_name in dataset_names:
        motifA, dimsA = test_lama(dataset_name, plot=plot)
        motifB, dimsB = test_lama(dataset_name, plot=plot, minimize_pairwise_dist=True)
        motifC, dimsC = test_mstamp(dataset_name, plot=plot, use_mdl=True)
        motifD, dimsD = test_mstamp(dataset_name, plot=plot, use_mdl=False)
        motifE, dimsE = test_emd_pca(dataset_name, plot=plot)
        motifF, dimsF = test_kmotifs(dataset_name, first_dims=True, plot=plot)
        motifG, dimsG = test_kmotifs(dataset_name, first_dims=False, plot=plot)

        method_names_dims = [name + "_dims" for name in method_names]
        columns = ["dataset", "k"]
        columns.extend(method_names)
        columns.extend(method_names_dims)
        df = pd.DataFrame(columns=columns)

        for i, k in enumerate(ks):
            df.loc[len(df.index)] \
                = [dataset_name, k,
                   motifA[i].tolist(), motifB[i].tolist(), motifC[0], motifD[0],
                   motifE[i].tolist(), motifF[i].tolist(), motifG[i].tolist(),
                   dimsA[i].tolist(), dimsB[i].tolist(), dimsC[0].tolist(),
                   dimsD[0].tolist(),
                   dimsE[i].tolist(), dimsF[i].tolist(), dimsG[i].tolist()]

        print("--------------------------")

        # from datetime import datetime
        # currentDateTime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        df.to_parquet(
            f'results/results_physio_{dataset_name}.gzip',  # _{currentDateTime}
            compression='gzip')


def test_plot_results():
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
        "K-Motifs (all)"
    ]

    results = []

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        df, ground_truth = load_physiodata()

        df_loc = pd.read_parquet(
            f"results/results_physio_{dataset_name}.gzip")

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

        pd.DataFrame(
            data=np.array(results),
            columns=["Dataset", "Method", "Precision", "Recall"]).to_csv(
            "results/physio_precision.csv")

        print(results)

        if True:
            plot_names = [
                "mSTAMP+MDL",
                "mSTAMP",
                "EMD*",
                "K-Motifs (all)",
                "LAMA",
            ]

            positions = [method_names.index(name) for name in plot_names]
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
