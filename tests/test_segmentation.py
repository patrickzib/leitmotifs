from leitmotifs.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import warnings

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

warnings.simplefilter("ignore")


def test_har():
    har_series = \
        pd.read_csv("../datasets/experiments/student_commute.txt",
                    delimiter="\t", header=None)[0]
    # ds_name = "student commute"

    cps = [0, 2012, 5662, 6800, 8795, 9712, 10467, 17970, 18870, 24169, 25896, 26754,
           27771, 33952, 34946, 43423, 43830, 47661, 56162, 56294, 56969, 57479, 58135]

    activities = ['start', 'walk', 'climb stairs', 'walk', 'go down stairs', 'walk',
                  'wait', 'get on', 'ride train (standing)', 'get off', 'walk',
                  'go down stairs', 'walk', 'wait for traffic lights', 'walk',
                  'wait for traffic lights', 'jog', 'walk fast', 'climb stairs', 'walk',
                  'climb stairs', 'walk', 'wait', 'end']

    for i, (a, b) in enumerate(zip(cps[:-1], cps[1:])):
        series = har_series.iloc[a:b].values
        ml = LAMA(ds_name=activities[i], series=series,
                  elbow_deviation=1.25, slack=0.6)

        k_max = 50
        length_range = np.arange(40, 150, 2)

        _ = ml.fit_motif_length(k_max, length_range, plot=False, subsample=2)

        ml.plot_motifset()

        if i > 3:
            break
