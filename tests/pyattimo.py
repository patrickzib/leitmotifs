import pyattimo
import pandas as pd
from motiflets.plotting import *

DATASETS = {
    "astro": "https://figshare.com/ndownloader/files/36982360",
    "ecg": "https://figshare.com/ndownloader/files/36982384",
    "freezer": "https://figshare.com/ndownloader/files/36982390",
    "gap": "https://figshare.com/ndownloader/files/36982396"
}


def load_dataset(dataset, prefix=None):
    import numpy
    from urllib.request import urlretrieve
    import os

    outfname = dataset + ".csv.gz"
    if not os.path.isfile(outfname):
        print("Downloading dataset")
        urlretrieve(DATASETS[dataset], outfname)

    if prefix is not None:
        return numpy.loadtxt(outfname)[:prefix]
    else:
        return numpy.loadtxt(outfname)


def test_multivariate():
    # load a dataset, any list of numpy array of floats works fine
    # The following call loads the first 100000 points of the ECG
    # dataset (which will be downloaded from the internet)
    ts = pyattimo.load_dataset('ecg', 30000)

    # Now we can find k-motiflets:
    #  - w is the window length
    #  - support is the number of subsequences in the motiflet (k in the motiflet paper)
    #  - repetitions is the number of LSH repetitions
    m = pyattimo.motiflet(ts, w=1000, support=10, repetitions=512)
    print(m.extent)


def test_motiflets():
    ts = load_dataset("ecg", 30000)
    series = pd.DataFrame(ts).T
    series

    ml = Motiflets("ECG", series)
    ml.fit_k_elbow(10, 1000)
