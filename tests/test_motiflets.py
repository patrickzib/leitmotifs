import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import subprocess
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import warnings

warnings.simplefilter("ignore")

import time


def test_har():
    har_series = pd.read_csv(
        "../datasets/experiments/student_commute.txt", delimiter="\t", header=None)[0]
    # ds_name = "student commute"

    cps = [0, 2012, 5662, 6800, 8795, 9712, 10467, 17970, 18870, 24169,
           25896, 26754, 27771, 33952, 34946, 43423, 43830, 47661,
           56162, 56294, 56969, 57479, 58135]

    activities = ['start', 'walk', 'climb stairs', 'walk', 'go down stairs', 'walk',
                  'wait', 'get on', 'ride train (standing)', 'get off', 'walk',
                  'go down stairs', 'walk', 'wait for traffic lights', 'walk',
                  'wait for traffic lights', 'jog', 'walk fast', 'climb stairs',
                  'walk', 'climb stairs', 'walk', 'wait', 'end']

    for i, (a, b) in enumerate(zip(cps[:-1], cps[1:])):
        series = har_series.iloc[a:b]
        ml = Motiflets(ds_name=activities[i], series=series,
                       elbow_deviation=1.25, slack=0.6)

        k_max = 50
        length_range = np.arange(40, 150, 1)

        motif_length, all_minima = ml.fit_motif_length(k_max, length_range, subsample=2)

        # motif_length = 50
        ml.fit_k_elbow(
            k_max, motif_length=motif_length,
            plot_elbows=False, plot_motifs_as_grid=True
        )

        # ml.plot_motifset()

        if i > 3:
            break


def test_parallel_distances():
    har_series = pd.read_csv(
        "../datasets/experiments/student_commute.txt", delimiter="\t", header=None)[0]
    # ds_name = "student commute"

    cps = [0, 20000, 40000]
    print("-----")

    for i, (a, b) in enumerate(zip(cps[:-1], cps[1:])):
        series = har_series[a:b].values

        start = time.time()
        D1 = ml.compute_distances_full_seq(series, 100)
        seq_end = time.time() - start

        start = time.time()
        D2 = ml.compute_distances_full(series, 100, n_jobs=4)
        par_end = time.time() - start

        # start = time.time()
        # ml.compute_distances_full(series, 100, n_jobs=8)
        # par2_end = time.time() - start

        print("Equal", np.allclose(D1, D2))
        print("Seq", seq_end,
              "\tPar (n_jobs=4)", par_end,
              # "\tPar (n_jobs=8)", par2_end
              )
