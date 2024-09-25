import stumpy
import os
import scipy

from leitmotifs.plotting import *
from numba import njit


def load_smm_results(
        series,
        ds_name,
        ground_truth,
        plot=True
):

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

    i = dataset_names.index(ds_name) + 1
    file = "../tests/results/smm_benchmark/results/1/Motif_"+str(i)+"_DepO_2_DepT_2.mat"
    if not os.path.exists(file):
        print(f"The file {file} does not exist.")
        return

    print(dataset_names[i - 1])

    mat_file = scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
    motif_bag = mat_file["MotifBag"]

    if not isinstance(motif_bag, np.ndarray):
        motif_bag = [motif_bag]

    best_f_score = 0.0
    best_motif_set = []
    best_dims = []
    best_length = 0
    precision, recall = 0, 0

    for motif_bag in motif_bag:
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
                np.sort(motif_set), ground_truth.values[0, 0], length)

            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            if f_score > best_f_score:
                best_f_score = f_score
                best_motif_set = motif_set
                best_length = length
                best_dims = dims

    if len(best_motif_set) > 0:
        if best_length == 1:
            best_length = 5
        print("\t", best_motif_set)
        print("\t", best_dims)
        print("\t", best_length)

        if plot:
            _, znormed_euclidean_distance = plot_motifsets(
                dataset_names[i - 1],
                series,
                motifsets=np.array([best_motif_set]),
                motifset_names=["SMM"],
                leitmotif_dims=np.array([best_dims]),
                motif_length=best_length,
                ground_truth=ground_truth,
                show=True)

    return np.array([best_motif_set]), np.array([best_dims])


def run_mstamp(df, ds_name, motif_length,
               ground_truth=None, plot=True,
               use_mdl=True, use_dims=None):
    series = df.values.astype(np.float64)

    # Find the Pair Motif
    mps, indices = stumpy.mstump(series, m=motif_length)
    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]
    mdls, subspaces = stumpy.mdl(series, motif_length, motifs_idx, nn_idx)

    if use_mdl:
        # Find the optimal dimensionality by minimizing the MDL
        k = np.argmin(mdls)
    else:
        # Use a pre-defined dimensionality
        k = use_dims - 1

    if plot and use_mdl:
        plt.plot(np.arange(len(mdls)), mdls, c='red', linewidth='2')
        plt.xlabel('k (zero-based)')
        plt.ylabel('Bit Size')
        plt.xticks(range(mps.shape[0]))
        plt.tight_layout()
        plt.show()

    print("Best dimensions", df.index[subspaces[k]])

    # found Pair Motif
    motif = [motifs_idx[subspaces[k]], nn_idx[subspaces[k]]]
    print("Pair Motif Position:")
    print("\tpos:\t", motif)
    print("\tf:  \t", subspaces[k])

    dims = np.array([subspaces[k]])
    motifs = np.array([[motifs_idx[subspaces[k]][0], nn_idx[subspaces[k]][0]]])
    motifset_names = ["mStamp"]

    if plot:
        _ = plot_motifsets(
            ds_name,
            df,
            motifsets=motifs,
            leitmotif_dims=dims,
            motifset_names=motifset_names,
            motif_length=motif_length,
            ground_truth=ground_truth,
            show=True)

    return motifs, dims


@njit(cache=True, fastmath=True)
def filter_non_trivial_matches(motif_set, m, slack=0.5):
    # filter trivial matches
    non_trivial_matches = []
    last_offset = - m
    for offset in np.sort(motif_set):
        if offset > last_offset + m * slack:
            non_trivial_matches.append(offset)
            last_offset = offset

    return np.array(non_trivial_matches)


def run_kmotifs(
        series,
        ds_name,
        motif_length,
        r_ranges,
        use_dims,
        target_k,
        slack=0.5,
        ground_truth=None,
        plot=True):
    D_full = ml.compute_distances_full_univ(
        series.iloc[:use_dims].values, motif_length, slack=slack)
    D_full = D_full.squeeze() / use_dims

    last_cardinality = 0
    for r in r_ranges:
        cardinality = -1
        k_motif_dist_var = -1
        motifset = []
        for order, dist in enumerate(D_full):
            motif_set = np.argwhere(dist <= r).flatten()
            if len(motif_set) > cardinality:
                # filter trivial matches
                motif_set = filter_non_trivial_matches(motif_set, motif_length, slack)

                # Break ties by variance of distances
                dist_var = np.var(dist[motif_set])
                if len(motif_set) > cardinality or \
                        (dist_var < k_motif_dist_var and len(motif_set) == cardinality):
                    cardinality = len(motif_set)
                    motifset = motif_set
                    k_motif_dist_var = dist_var

        if cardinality != last_cardinality:
            # print(f"cardinality: {cardinality} for r={r}")
            last_cardinality = cardinality

        if cardinality >= target_k:
            print(f"Radius: {r}, K: {cardinality}")
            # print(f"Pos: {motifset}")
            motifset_names = ["K-Motif"]

            if plot:
                plot_motifsets(
                    ds_name,
                    series,
                    motifsets=motifset.reshape(1, -1),
                    leitmotif_dims=np.arange(use_dims).reshape(1, -1),
                    motifset_names=motifset_names,
                    motif_length=motif_length,
                    ground_truth=ground_truth,
                    show=True)

            return motifset, use_dims

    return [], []


def compute_precision_recall(pred, gt, motif_length):
    if motif_length == 0:
       return 0, 0

    gt_found = np.zeros(len(gt))
    pred_correct = np.zeros(len(pred))
    for a, start in enumerate(pred):
        for i, g_start in enumerate(gt):
            end = start + motif_length
            length_interval1 = end - start
            length_interval2 = g_start[1] - g_start[0]

            # Calculate overlapping portion
            overlap_start = max(start, g_start[0])
            overlap_end = min(end, g_start[1])
            overlap_length = max(0, overlap_end - overlap_start)

            if overlap_length >= 0.5 * min(length_interval1, length_interval2):
                gt_found[i] = 1
                pred_correct[a] = 1

    return np.average(pred_correct), np.average(gt_found)


def run_tests(
        dataset_name,
        ks,
        method_names,
        test_lama,    # function
        test_mstamp,  # function
        test_emd_pca, # function
        test_kmotifs, # function
        file_prefix,
        test_smm=None,     # function
        plot=False,
      ):

    motifs_list = []
    dims_list = []
    if "LAMA" in method_names:
        motifA, dimsA = test_lama(dataset_name, plot=plot)
        motifs_list.append(motifA)
        dims_list.append(dimsA)
    if "LAMA (naive)" in method_names:
        motifB, dimsB = test_lama(dataset_name, plot=plot, minimize_pairwise_dist=True)
        motifs_list.append(motifB)
        dims_list.append(dimsB)
    if "mSTAMP+MDL" in method_names:
        motifC, dimsC = test_mstamp(dataset_name, plot=plot, use_mdl=True)
        motifs_list.append(motifC)
        dims_list.append(dimsC)
    if "mSTAMP" in method_names:
        motifD, dimsD = test_mstamp(dataset_name, plot=plot, use_mdl=False)
        motifs_list.append(motifD)
        dims_list.append(dimsD)
    if "EMD*" in method_names:
        motifE, dimsE = test_emd_pca(dataset_name, plot=plot)
        motifs_list.append(motifE)
        dims_list.append(dimsE)
    if "K-Motifs (TOP-f)" in method_names:
        motifF, dimsF = test_kmotifs(dataset_name, first_dims=True, plot=plot)
        motifs_list.append(motifF)
        dims_list.append(dimsF)
    if "K-Motifs (all)" in method_names:
        motifG, dimsG = test_kmotifs(dataset_name, first_dims=False, plot=plot)
        motifs_list.append(motifG)
        dims_list.append(dimsG)
    if "SMM" in method_names:
        motifX, dimsX = test_smm(dataset_name, plot=plot)
        motifs_list.append(motifX)
        dims_list.append(dimsX)

    # Distances
    if "LAMA (cid)" in method_names:
        motifH, dimsH = test_lama(dataset_name, plot=plot, distance="cid")
        motifs_list.append(motifH)
        dims_list.append(dimsH)
    if "LAMA (ed)" in method_names:
        motifI, dimsI = test_lama(dataset_name, plot=plot, distance="ed")
        motifs_list.append(motifI)
        dims_list.append(dimsI)
    if "LAMA (cosine)" in method_names:
        motifJ, dimsJ = test_lama(dataset_name, plot=plot, distance="cosine")
        motifs_list.append(motifJ)
        dims_list.append(dimsJ)

    # Exclusion Zones
    if "LAMA (alpha=0)" in method_names:
        motifJ, dimsJ = test_lama(dataset_name, plot=plot, exclusion_range=0.0)
        motifs_list.append(motifJ)
        dims_list.append(dimsJ)
    if "LAMA (alpha=0.25)" in method_names:
        motifJ, dimsJ = test_lama(dataset_name, plot=plot, exclusion_range=0.25)
        motifs_list.append(motifJ)
        dims_list.append(dimsJ)
    if "LAMA (alpha=0.5)" in method_names:
        motifJ, dimsJ = test_lama(dataset_name, plot=plot, exclusion_range=0.50)
        motifs_list.append(motifJ)
        dims_list.append(dimsJ)
    if "LAMA (alpha=0.75)" in method_names:
        motifJ, dimsJ = test_lama(dataset_name, plot=plot, exclusion_range=0.75)
        motifs_list.append(motifJ)
        dims_list.append(dimsJ)
    if "LAMA (alpha=1)" in method_names:
        motifJ, dimsJ = test_lama(dataset_name, plot=plot, exclusion_range=1.0)
        motifs_list.append(motifJ)
        dims_list.append(dimsJ)

    method_names_dims = [name + "_dims" for name in method_names]
    columns = ["dataset", "k"]
    columns.extend(method_names)
    columns.extend(method_names_dims)
    df = pd.DataFrame(columns=columns)

    for i, k in enumerate(ks):
        motif_sets = []
        motif_dims = []
        for j in range(len(motifs_list)):
            if len(motifs_list[j]) > i:
                # if there are multiple motifs
                motif_sets.append(motifs_list[j][i].tolist())
                motif_dims.append(dims_list[j][i].tolist())
            else:
                # if there is only one motif
                motif_sets.append(motifs_list[j][0].tolist())
                motif_dims.append(dims_list[j][0].tolist())

        # concatenate the three lists
        df.loc[len(df.index)] = [dataset_name, k] + motif_sets + motif_dims

    print("--------------------------")

    # from datetime import datetime
    df.to_parquet(
        f'results/{file_prefix}_{dataset_name}.gzip', compression='gzip')


def eval_tests(
        dataset_name,
        ds_name,
        df,
        method_names,
        motif_length,
        ground_truth,
        all_plot_names,
        file_prefix,
        results,
        plot=True):
    df_loc = pd.read_parquet(f"results/{file_prefix}_{dataset_name}.gzip")

    motifs = []
    dims = []
    for id in range(df_loc.shape[0]):
        for motif_method in method_names:
            motifs.append(df_loc.loc[id][motif_method])
            dims.append(df_loc.loc[id][motif_method + "_dims"])

    # write results to file
    for id in range(df_loc.shape[0]):
        for method, motif_set in zip(
                method_names,
                motifs[id * len(method_names): (id + 1) * len(method_names)]
        ):
            precision, recall = compute_precision_recall(
                np.sort(motif_set), ground_truth.values[0, 0], motif_length)
            results.append([ds_name, method, precision, recall])

    if plot:
        for plot_name in all_plot_names:
            plot_names = all_plot_names[plot_name]
            positions = [method_names.index(name) for name in plot_names]
            out_path = "results/images/" + dataset_name + plot_name + ".pdf"
            plot_motifsets(
                ds_name,
                df,
                motifsets=[motifs[pos] for pos in positions],
                leitmotif_dims=[dims[pos] for pos in positions],
                motifset_names=plot_names,
                motif_length=motif_length,
                ground_truth=ground_truth,
                show=out_path is None)

            if out_path is not None:
                plt.savefig(out_path)
                plt.show()