import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import subprocess
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import warnings
warnings.simplefilter("ignore")



def test_har():
    har_series = pd.read_csv(
        "../datasets/experiments/student_commute.txt", delimiter="\t", header=None)[0]
    ds_name = "student commute"

    cps = [0, 2012,5662,6800,8795,9712,10467,17970,18870,24169,
           25896,26754,27771,33952,34946,43423,43830,47661,
           56162,56294,56969,57479,58135]
    activities = ['start','walk','climb,stairs','walk','go down stairs','walk','wait',
                  'get,on','ride, train (standing)','get,off','walk','go down stairs',
                  'walk','wait for traffic lights','walk','wait for traffic lights',
                  'jog','walk fast','climb stairs','walk','climb stairs','walk','wait','end']

    for i, (a, b) in enumerate(zip(cps[:-1], cps[1:])):
        series = har_series[a:b].values
        ks = 50

        length_range = np.arange(20,150,10)
        # motif_length = plot_motif_length_selection(
        #    ks, series, length_range, activities[i]
        # )

        motif_length = 50
        dists, motiflets, elbow_points = plot_elbow(
            ks, series, ds_name=activities[i],  # , plot_elbows=True,
            motif_length=motif_length)