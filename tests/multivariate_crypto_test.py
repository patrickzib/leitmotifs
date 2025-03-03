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
    # ada_eur = pd.read_csv("../datasets/crypto/ADA-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # bitcoin_usd = pd.read_csv("../datasets/crypto/BTC-USD.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # bitcoin_gbp = pd.read_csv("../datasets/crypto/BTC-GBP.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # bitcoin_eur = pd.read_csv("../datasets/crypto/BTC-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # bitcoin_cash = pd.read_csv("../datasets/crypto/BCH-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # ethereum = pd.read_csv("../datasets/crypto/ETH-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # litecoin = pd.read_csv("../datasets/crypto/LTC-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # solana = pd.read_csv("../datasets/crypto/SOL-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    # xrp = pd.read_csv("../datasets/crypto/XRP-EUR.csv").set_index("Date")[
    #     ["Close", "Volume"]]
    #
    # ada_eur.index = pd.to_datetime(ada_eur.index)
    # bitcoin_usd.index = pd.to_datetime(bitcoin_usd.index)
    # bitcoin_gbp.index = pd.to_datetime(bitcoin_gbp.index)
    # bitcoin_eur.index = pd.to_datetime(bitcoin_eur.index)
    # bitcoin_cash.index = pd.to_datetime(bitcoin_cash.index)
    # ethereum.index = pd.to_datetime(ethereum.index)
    # litecoin.index = pd.to_datetime(litecoin.index)
    # solana.index = pd.to_datetime(solana.index)
    # xrp.index = pd.to_datetime(xrp.index)
    #
    # ada_eur["Name"] = "Cardano"
    # bitcoin_usd["Name"] = "Bitcoin (USD)"
    # bitcoin_gbp["Name"] = "Bitcoin (GBP)"
    # bitcoin_eur["Name"] = "Bitcoin (EUR)"
    # bitcoin_cash["Name"] = "Bitcoin Cash"
    # ethereum["Name"] = "Ethereum"
    # litecoin["Name"] = "Litecoin"
    # solana["Name"] = "Solana"
    # xrp["Name"] = "XRP"
    #
    # for df_apply in [ada_eur, bitcoin_usd, bitcoin_gbp, bitcoin_eur, bitcoin_cash,
    #                  ethereum, litecoin, solana, xrp]:
    #     df_apply[["Close"]] = df_apply[["Close"]].apply(np.log2).apply(normalize)
    #     df_apply[["Volume"]] = df_apply[["Volume"]].apply(normalize)
    #
    # df = pd.concat([ada_eur, bitcoin_cash, bitcoin_eur, ethereum, litecoin, solana,
    #                 xrp])  # bitcoin_gbp, bitcoin_usd,
    # df["Name"] = df["Name"].astype("category")
    #
    # df_pivot = df.pivot(columns="Name", values="Close").fillna(method="bfill").T
    # # df_pivot.to_csv("../datasets/crypto/crypto.csv", index=False)

    df = pd.read_csv("../datasets/crypto/crypto.csv")
    df.columns = pd.to_datetime(df.columns)
    df_gt = read_ground_truth("../datasets/crypto/crypto")

    return df, df_gt


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
        distance="znormed_ed",
        exclusion_range=None,
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

    ml = LAMA(
        ds_name, df_transform,
        dimension_labels=df.index,
        distance=distance,
        n_dims=n_dims,
        ground_truth=ground_truth,
        minimize_pairwise_dist=minimize_pairwise_dist,
        slack=exclusion_range if exclusion_range else slack,
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
            r_ranges=np.arange(1, 200, 1),
            use_dims=n_dims if first_dims else df.shape[0],  # first dims or all dims
            target_k=target_k,
            ground_truth=ground_truth,
            plot=plot
        )
        used_dims.append(np.arange(dims))
        motif_sets.append(motif)

    return motif_sets, used_dims


def test_publication(plot=False, method_names=None):
    dataset_names = [
        "Bitcoin-Halving"
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

    file_prefix = "results_stocks"

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


def test_plot_results(plot=True, method_names=None, all_plot_names=None):
    dataset_names = [
        "Bitcoin-Halving"
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

    file_prefix = "results_stocks"
    output_file = "stocks_precision"

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        df, ground_truth = load_crypto()

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
        "results/" + output_file + ".csv")
