import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import subprocess
import matplotlib
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

    k_max = 20
    length_range = np.arange(40, 100, 10)
    motif_length = plot_motif_length_selection(
        k_max,
        data.values,
        length_range,
        "left-0"
    )
    print (motif_length)

    motif_length = 100
    dists, motiflets, elbow_points = plot_elbow(
        k_max, data,
        ds_name=ds_name,
        plot_elbows=True,
        motif_length=motif_length,
        slack=1.0,
        elbow_deviation=1.25,
        method_name="K-Motiflets")