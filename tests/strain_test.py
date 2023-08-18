import matplotlib

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import warnings

warnings.simplefilter("ignore")


def test_strain():
    ds_name = "Strain"

    data = pd.read_csv("../datasets/experiments/strain-0-left.csv",
                       index_col=0, squeeze=True)
    print("Dataset Original Length n: ", len(data))
    data, factor = ml._resample(data, sampling_factor=10000)
    data[:] = zscore(data)

    ml2 = Motiflets(ds_name=ds_name, series=data,
                    elbow_deviation=1.25, slack=1.0)

    k_max = 20
    length_range = np.arange(40, 100, 1)

    motif_length, all_minima = ml2.fit_motif_length(k_max, length_range)
    print(motif_length)

    motif_length = 100
    ml2.fit_k_elbow(
        k_max, motif_length=motif_length,
        plot_elbows=True)
