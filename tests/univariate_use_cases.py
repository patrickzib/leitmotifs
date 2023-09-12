import motiflets.motiflets as mof
from motiflets.plotting import *


def test_ecg():
    file = 'ecg-heartbeat-av.csv'
    ds_name = "ECG Heartbeat"
    series, df_gt = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df,
                   # elbow_deviation=1.25,
                   # slack=0.25,
                   ground_truth=df_gt
                   )

    ml.plot_dataset()

    k_max = 20
    motif_length_range = np.arange(50, 180, 5)
    best_length, all_minima = ml.fit_motif_length(
        k_max, motif_length_range, subsample=1)

    print("Best found length", best_length)

    exclusion = ml.motiflets[ml.elbow_points[-1]]
    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        exclusion=exclusion,  # TODO: refactor?
        exclusion_length=best_length,
        subsample=1,
        plot=True,
        plot_elbows=True,
        plot_motifs_as_grid=True
    )

    print("Best found length", best_length)


def test_vanilla_ice():
    file = 'vanilla_ice.csv'
    ds_name = "Vanilla Ice"
    series, df_gt = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df,
                   # elbow_deviation=1.25,
                   # slack=0.25,
                   ground_truth=df_gt
                   )
    ml.plot_dataset()
    k_max = 20
    length_range = np.arange(50, 350, 25)
    best_length, all_minima = ml.fit_motif_length(k_max, length_range, subsample=1)

    print("Best found length", best_length)

    exclusion = ml.motiflets[ml.elbow_points[-1]]
    best_length, _ = ml.fit_motif_length(
        k_max,
        length_range,
        exclusion=exclusion,  # TODO: refactor?
        exclusion_length=best_length,
        subsample=1,
        plot=True,
        plot_elbows=True,
        plot_motifs_as_grid=True
    )

    print("Best found length", best_length)


def test_muscle_activation():
    file = 'muscle_activation.csv'
    ds_name = "Muscle Activation"

    series, df_gt = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df,
                   ground_truth=df_gt,
                   # elbow_deviation=1.25,
                   slack=0.4
                   )

    ml.plot_dataset()

    k_max = 20
    length_range = np.arange(400, 700, 25)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    print("Best found length", best_motif_length)

    exclusion = ml.motiflets[ml.elbow_points[-1]]
    best_length, _ = ml.fit_motif_length(
        k_max,
        length_range,
        exclusion=exclusion,  # TODO: refactor?
        exclusion_length=best_motif_length,
        subsample=1,
        plot=True,
        plot_elbows=True,
        plot_motifs_as_grid=True
    )

    print("Best found length", best_length)


def test_physiodata():
    """ Funktioniert: 2 Motifs variabler Länge """
    file = 'npo141.csv'  # Dataset Length n:  269286
    ds_name = "EEG Sleep Data"
    series = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    k_max = 50
    length_range = np.arange(10, 100, 5)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    print("Best found length", best_motif_length)

    exclusion = ml.motiflets[ml.elbow_points[-1]]
    best_length, _ = ml.fit_motif_length(
        k_max,
        length_range,
        exclusion=exclusion,  # TODO: refactor?
        exclusion_length=best_motif_length,
        subsample=1,
        plot=True,
        plot_elbows=True,
        plot_motifs_as_grid=True
    )

def test_winding():
    file = "winding_col.csv"
    ds_name = "Industrial winding process"
    series = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    k_max = 12
    length_range = np.arange(50, 100, 1)

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()


def test_fnirs():
    file = "fNIRS_subLen_600.csv"
    ds_name = "fNIRS"
    series = mof.read_dataset_with_index(file)
    df = pd.DataFrame(series).T

    k_max = 20
    length_range = np.arange(20, 200, 5)

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()


def test_insects():
    ds_name = "Insect"
    data = pd.read_csv("../datasets/experiments/insect.csv",
                       squeeze=True)
    print("Dataset Original Length n: ", len(data))
    data, factor = mof._resample(data[:100000], sampling_factor=25000)
    data[:] = zscore(data)
    df = pd.DataFrame(data).T

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    k_max = 10
    length_range = np.arange(25, 125, 2)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=2)

    # for motif_length in length_range[all_minima]:
    #     ml.fit_k_elbow(k_max, motif_length,
    #                    plot_elbows=False,
    #                    plot_motifs_as_grid=True)
    #
    #     # ml.plot_motifset()


def test_gait():
    ds_name = "Gait"
    data = pd.read_csv("../datasets/experiments/human_gait.txt",
                       squeeze=True, header=None)
    print("Dataset Original Length n: ", data.shape)
    data[:] = zscore(data)
    df = pd.DataFrame(data).T.iloc[[1, 2, 5]]

    ml = Motiflets(ds_name=ds_name, series=df)
    ml.plot_dataset()

    k_max = 10
    length_range = np.arange(25, 50, 1)
    best_motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range, subsample=1)
