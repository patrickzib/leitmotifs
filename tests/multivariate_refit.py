import os
from leitmotifs.plotting import *

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

# path outside the git
path = "../../motiflets_use_cases/refit/"


def read_refit_data(house_index=1):
    file = path + "House" + str(house_index) + ".csv"
    if os.path.isfile(file):
        series = pd.read_csv(file,
                             names=(["UNIX TIMESTAMP (UCT)",
                                     "Aggregate",
                                     "Appliance1",
                                     "Appliance2",
                                     "Appliance3",
                                     "Appliance4",
                                     "Appliance5",
                                     "Appliance6",
                                     "Appliance7",
                                     "Appliance8",
                                     "Appliance9"]),
                             delimiter=",", header=None)
        series.set_index("UNIX TIMESTAMP (UCT)", inplace=True)
        series.index = pd.to_datetime(series.index, unit='s', origin="unix")

        # 5 minute intervals
        series = series.resample("300S").mean()
        series.ffill(inplace=True)
        series.bfill(inplace=True)

        mask = ((series.index.month == 4)
                & (series.index.year == 2014)
                & (series.index.day < 15))
        # print(series.loc[mask].T)
        ds_name = "REFIT Dataset"
        print("Shape", series.loc[mask].index)
        return ds_name, series.loc[mask].T
    else:
        raise ("File not found")


def test_plot_data():
    ds_name, series = read_refit_data()
    ml = LAMA(ds_name, series)
    ml.plot_dataset()


def test_multivariate():
    ds_name, series = read_refit_data(house_index=10)

    ml = LAMA(ds_name,
              series,
              n_dims=2
              )

    k_max = 20
    motif_length_range = np.arange(20, 350, 10)

    best_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=False,
        plot_motifsets=True,
        plot_best_only=False
    )
    # ml.plot_motifset()

    # dists[a] = ml.dists[ml.elbow_points[-1]]
    print("Best found length", best_length)
