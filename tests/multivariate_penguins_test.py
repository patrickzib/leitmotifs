import matplotlib
import scipy.cluster.hierarchy as sch
import scipy.io as sio

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/chains/"


def test_plot_data():
    series = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    series = series.iloc[497699-5000: 497699+5000, np.array([0, 7])].T

    plot_dataset(ds_name, series)


def test_univariate():
    test = sio.loadmat(path + 'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T

    ds_name = "Penguins (Snippet)"
    plot_dataset(ds_name, series)

    ks = 50
    motif_length_range = np.arange(10, 35, 1)

    best_motif_length, all_minima = plot_motif_length_selection(
        ks, series, motif_length_range, ds_name,
        elbow_deviation=1.25,
        subsample=1)

    # motif_length = 22

    for minimum in all_minima:
        motif_length = motif_length_range[minimum]
        dists, motiflets, elbow_points = plot_elbow(
            ks, series,
            ds_name=ds_name,
            plot_elbows=False,
            plot_grid=False,
            motif_length=motif_length,
            elbow_deviation=1.25
        )

        plot_motifset(
           ds_name,
           series,
           motifset=motiflets[elbow_points[-1]],
           dist=dists[elbow_points[-1]],
           motif_length=motif_length,
           show=True)


def test_multivariate():
    length = 1000
    B = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    for start in [0, 2000]:
        series = B.iloc[497699 + start:497699 + start + length, np.array([0, 1, 2, 3])].T
        # plot_dataset(ds_name, series)

        ks = 50
        motif_length_range = np.arange(10, 35, 1)

        best_motif_length, all_minima = plot_motif_length_selection(
            ks, series, motif_length_range, ds_name,
            elbow_deviation=1.1,
            subsample=1
        )

        # ks = 60
        motif_length = 22
        dists, motiflets, elbow_points = plot_elbow(
            ks, series,
            ds_name=ds_name,
            plot_elbows=True,
            plot_grid=False,
            motif_length=motif_length,
            # slack=0.6,
            elbow_deviation=1.1
        )

        fig, ax = plot_motifset(
            ds_name,
            series,
            motifset=motiflets[elbow_points[-1]],
            dist=dists[elbow_points[-1]],
            motif_length=motif_length,
            show=True)

        #plt.savefig(
        #   "penguin/" + ds_name + "_start_" + str(start) + ".pdf")


def test_dimension_plotting():
    length = 1000
    B = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    df = B.iloc[497699: 497699 + length, 0:9].T

    ks = 60
    motif_length = 22

    dists, motiflets, elbow_points = plot_elbow_by_dimension(
        ks, df,
        dimension_labels=df.index,
        ds_name=ds_name,
        motif_length=motif_length)

    series = np.zeros((df.shape[0], df.shape[1] - motif_length),
                      dtype=np.float32)
    for i in range(series.shape[0]):
        for pos in motiflets[i, elbow_points[i][-1]]:
            series[i, pos:pos + motif_length] = 1

    X = series

    # size of image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    Z = sch.linkage(X, method='ward')

    # creating the dendrogram
    dend = sch.dendrogram(
        Z, labels=df.index, ax=ax)

    plt.axhline(y=127.5, color='orange')
    ax.set_title('Dendrogram')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Euclidean distances')
    plt.tight_layout()
    plt.show()

    k = 2
    y_dimensions = sch.fcluster(Z, k, criterion='maxclust')
    mapping = list(zip(y_dimensions, df.index))

    joint_clusters = {}
    for i in range(1, k + 1):
        print("Cluster", i)
        joint_clusters[i] = [x[1] for x in mapping if x[0] == i]
        print(joint_clusters[i])
        print("----")


def test_dimension_plotting_top_down():
    length = 1000
    B = pd.read_csv(path + "penguin.txt", delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    df = B.iloc[497699: 497699 + length, 0:9].T

    ks = 60
    motif_length = 22

    dists, motiflets, elbow_points = ml.search_multidim_k_motiflets_elbow_top_down(
        ks, df, motif_length)
