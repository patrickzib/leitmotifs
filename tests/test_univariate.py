from leitmotifs.lama import *
from leitmotifs.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150


def test_univariate():
    file = 'ecg-heartbeat-av.csv'
    ds_name = "ECG Heartbeat"
    series = read_dataset_with_index(file)

    ml = LAMA(ds_name, series)
    ml.plot_dataset()

    ks = 20
    length_range = np.arange(25, 200, 25)
    ml.fit_motif_length(ks, length_range)
    print("Best motif length", series.index[ml.motif_length], "seconds")

    ml.fit_k_elbow(ks, ml.motif_length)
