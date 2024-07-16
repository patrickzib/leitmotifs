import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

from leitmotifs.competitors import *
from leitmotifs.lama import *


def normalize(x):
    std = np.std(x)
    mean = np.mean(x)
    if std == 0:
        return x - mean
    return (x - mean) / std


def load_crypto():
    ada_eur = pd.read_csv("../datasets/crypto/ADA-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]
    bitcoin_usd = pd.read_csv("../datasets/crypto/BTC-USD.csv").set_index("Date")[
        ["Close", "Volume"]]
    bitcoin_gbp = pd.read_csv("../datasets/crypto/BTC-GBP.csv").set_index("Date")[
        ["Close", "Volume"]]
    bitcoin_eur = pd.read_csv("../datasets/crypto/BTC-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]
    bitcoin_cash = pd.read_csv("../datasets/crypto/BCH-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]
    ethereum = pd.read_csv("../datasets/crypto/ETH-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]
    litecoin = pd.read_csv("../datasets/crypto/LTC-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]
    solana = pd.read_csv("../datasets/crypto/SOL-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]
    xrp = pd.read_csv("../datasets/crypto/XRP-EUR.csv").set_index("Date")[
        ["Close", "Volume"]]

    ada_eur.index = pd.to_datetime(ada_eur.index)
    bitcoin_usd.index = pd.to_datetime(bitcoin_usd.index)
    bitcoin_gbp.index = pd.to_datetime(bitcoin_gbp.index)
    bitcoin_eur.index = pd.to_datetime(bitcoin_eur.index)
    bitcoin_cash.index = pd.to_datetime(bitcoin_cash.index)
    ethereum.index = pd.to_datetime(ethereum.index)
    litecoin.index = pd.to_datetime(litecoin.index)
    solana.index = pd.to_datetime(solana.index)
    xrp.index = pd.to_datetime(xrp.index)

    ada_eur["Name"] = "Cardano"
    bitcoin_usd["Name"] = "Bitcoin (USD)"
    bitcoin_gbp["Name"] = "Bitcoin (GBP)"
    bitcoin_eur["Name"] = "Bitcoin (EUR)"
    bitcoin_cash["Name"] = "Bitcoin Cash"
    ethereum["Name"] = "Ethereum"
    litecoin["Name"] = "Litecoin"
    solana["Name"] = "Solana"
    xrp["Name"] = "XRP"

    for df_apply in [ada_eur, bitcoin_usd, bitcoin_gbp, bitcoin_eur, bitcoin_cash,
                     ethereum, litecoin, solana, xrp]:
        df_apply[["Close"]] = df_apply[["Close"]].apply(np.log2).apply(normalize)
        df_apply[["Volume"]] = df_apply[["Volume"]].apply(normalize)

    df = pd.concat([ada_eur, bitcoin_cash, bitcoin_eur, ethereum, litecoin, solana,
                    xrp])  # bitcoin_gbp, bitcoin_usd,
    df["Name"] = df["Name"].astype("category")

    df_pivot = df.pivot(columns="Name", values="Close").fillna(method="bfill")

    df_gt = read_ground_truth("../datasets/crypto/crypto")

    return df_pivot.T, df_gt


datasets = {
    "Bitcoin-Halving": {
        "ks": [3],
        "motif_length": 180,
        "n_dims": 2,
        "slack": 1.0,
        # "length_range": np.arange(120, 240, 10),
    },
}


def get_ds_parameters(name="Bitcoin-Halving"):
    global ds_name, k_max, n_dims, length_range, motif_length
    global audio_file_url, pandas_file_url, ks, slack

    ds_name = name
    dataset = datasets[name]
    ks = dataset["ks"]
    k_max = np.max(ks) + 2
    n_dims = dataset["n_dims"]
    # length_range = dataset["length_range"]
    slack = dataset["slack"]
    motif_length = dataset["motif_length"]


def test_lama(
        dataset_name="Bitcoin-Halving",
        minimize_pairwise_dist=False,
        use_PCA=False,
        motifset_name="LAMA",
        plot=True):
    get_ds_parameters(dataset_name)
    df, ground_truth = load_crypto()

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


def test_emd_pca(dataset_name="Bitcoin-Halving", plot=True):
    return test_lama(dataset_name, use_PCA=True, motifset_name="PCA", plot=plot)


def test_mstamp(dataset_name="Bitcoin-Halving", plot=True, use_mdl=True):
    get_ds_parameters(dataset_name)
    df, ground_truth = load_crypto()

    return run_mstamp(df, ds_name, motif_length=motif_length,
                      ground_truth=ground_truth, plot=plot,
                      use_mdl=use_mdl, use_dims=n_dims)


def test_kmotifs(dataset_name="Bitcoin-Halving", first_dims=True, plot=True):
    get_ds_parameters(dataset_name)
    df, ground_truth = load_crypto()

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
        "Bitcoin-Halving"
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
            f'results/results_stocks_{dataset_name}.gzip',  # _{currentDateTime}
            compression='gzip')


def test_plot_results():
    dataset_names = [
        "Bitcoin-Halving"
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
        df, ground_truth = load_crypto()

        df_loc = pd.read_parquet(
            f"results/results_stocks_{dataset_name}.gzip")

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
            "results/stocks_precision.csv")

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
