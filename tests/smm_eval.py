import sys

sys.path.insert(0, "../")

import scipy.io

from leitmotifs.lama import *

from leitmotifs.competitors import *

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import warnings

warnings.simplefilter("ignore")

dataset_names = [
    'physio',
    'Boxing',
    'Swordplay',
    'Basketball',
    'Charleston - Side By Side Female',
    'crypto',
    'birds',
    "What I've Done - Linkin Park",
    'Numb - Linkin Park',
    'Vanilla Ice - Ice Ice Baby',
    'Queen David Bowie - Under Pressure',
    'The Rolling Stones - Paint It, Black',
    'Star Wars - The Imperial March',
    'Lord of the Rings Symphony - The Shire']

scores = {}
# Load the .mat file
for i in range(1, 15):

    series = pd.read_csv("../notebooks/smm_results/lama_benchmark/" + str(i) + ".csv", header=None).T
    file = "../notebooks/smm_results/lama_benchmark/Motif_" + str(i) + "_DepO_2_DepT_2.mat"
    df_gt = read_ground_truth("../notebooks/smm_results/lama_benchmark/" + str(i) + ".csv")

    """ only for plotting
    ml = LAMA(
        dataset_names[i-1],  
        series,
        dimension_labels=df.index,    
    )
    ml.plot_dataset()
    """

    # break
    # some dataset found no motifs
    if not os.path.exists(file):
        print(f"The file {file} does not exist.")
        continue

    print(dataset_names[i - 1])

    mat_file = scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
    motif_bags = mat_file["MotifBag"]

    if not isinstance(motif_bags, np.ndarray):
        motif_bags = [motif_bags]

    best_f_score = 0.0
    best_motif_set = []
    best_dims = []
    best_length = 0
    precision, recall = 0, 0

    for motif_bag in motif_bags:
        if motif_bag:
            startIdx = motif_bag.startIdx

            motif_set = startIdx
            dims = motif_bag.depd[0] - 1  # matlab uses 1-indexing but python 0-indexing
            if not isinstance(dims, np.ndarray):
                dims = [dims]

            length = motif_bag.Tscope[0]
            if length == 0:
                length = 1

            precision, recall = compute_precision_recall(
                np.sort(motif_set), df_gt.values[0, 0], length)

            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            if f_score > best_f_score:
                best_f_score = f_score
                best_motif_set = motif_set
                best_length = length
                best_dims = dims
                best_precision = precision
                best_recall = recall

    precision, recall = compute_precision_recall(
        np.sort(best_motif_set), df_gt.values[0, 0], best_length)

    scores[i] = [dataset_names[i - 1], precision, recall]
    print("\tMotifs:\t", len(motif_bags))
    print("\tDims:\t",best_dims)
    print("\tLength:\t",best_length)
    print("\t", scores[i])

df = pd.DataFrame(scores).T
df.columns = "Dataset Precision Recall".split()
df.to_csv('csv/smm_results.csv', index=False)