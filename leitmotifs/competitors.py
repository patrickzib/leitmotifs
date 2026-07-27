import stumpy
import os
import numpy as np
import pandas as pd
import scipy
import warnings
from pathlib import Path

from leitmotifs.plotting import *
from numba import njit


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def has_smm_motif_bag(motif_bag):
    if motif_bag is None:
        return False
    if isinstance(motif_bag, np.ndarray) and motif_bag.size == 0:
        return False
    return hasattr(motif_bag, "startIdx")


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
    file = (
        _PROJECT_ROOT
        / "tests"
        / "results"
        / "smm_benchmark"
        / "results"
        / "1"
        / f"Motif_{i}_DepO_2_DepT_2.mat"
    )
    if not file.exists():
        print(f"The file {file} does not exist.")
        return np.array([]), np.array([])

    print(f"Loading SMM results: {dataset_names[i - 1]}")

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
        if has_smm_motif_bag(motif_bag):
            startIdx = motif_bag.startIdx

            motif_set = startIdx
            dims = motif_bag.depd[0] - 1  # matlab uses 1-indexing but python 0-indexing
            if not isinstance(dims, np.ndarray):
                dims = [dims]

            length = motif_bag.Tscope[0]
            if length == 0:
                length = 1

            precision, recall = compute_best_precision_recall(
                np.sort(motif_set), ground_truth, length)

            f_score = compute_f_score(precision, recall)
            if f_score > best_f_score:
                best_f_score = f_score
                best_motif_set = motif_set
                best_length = length
                best_dims = dims

    if len(best_motif_set) > 0:
        if best_length == 1:
            best_length = 5
        print("SMM motif positions:", np.asarray(best_motif_set, dtype=np.int64).tolist())
        print("SMM dims:", np.asarray(best_dims, dtype=np.int64).reshape(-1).tolist())
        print("SMM motif length:", int(best_length))

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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'where' used without 'out'.*",
            category=UserWarning,
            module="stumpy.core",
        )
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

    selected_dims = np.asarray(subspaces[k]).reshape(-1)
    print("Best dimensions:", list(df.index[selected_dims]))

    # found Pair Motif
    motif = [motifs_idx[subspaces[k]], nn_idx[subspaces[k]]]
    motif_positions = [int(motifs_idx[dim]) for dim in selected_dims]
    nearest_neighbor_positions = [int(nn_idx[dim]) for dim in selected_dims]
    print("Pair motif positions:", list(zip(motif_positions, nearest_neighbor_positions)))
    print("Pair motif dims:", [int(dim) for dim in selected_dims])

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
                if len(motif_set) == 0:
                    continue

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

    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    gt = np.asarray(gt, dtype=np.int64)
    motif_length = int(motif_length)
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0

    gt_found = np.zeros(len(gt))
    pred_correct = np.zeros(len(pred))
    for a, start in enumerate(pred):
        for i, g_start in enumerate(gt):
            start = int(start)
            gt_start = int(g_start[0])
            gt_end = int(g_start[1])
            end = start + motif_length
            length_interval1 = end - start
            length_interval2 = gt_end - gt_start

            # Calculate overlapping portion
            overlap_start = max(start, gt_start)
            overlap_end = min(end, gt_end)
            overlap_length = max(0, overlap_end - overlap_start)

            if overlap_length >= 0.5 * min(length_interval1, length_interval2):
                gt_found[i] = 1
                pred_correct[a] = 1

    return np.average(pred_correct), np.average(gt_found)


def compute_f_score(precision, recall):
    precision = float(precision)
    recall = float(recall)
    if not np.isfinite(precision) or not np.isfinite(recall):
        return 0.0
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_best_precision_recall(pred, ground_truth, motif_length):
    best_precision = 0.0
    best_recall = 0.0
    best_f_score = 0.0

    if ground_truth is None:
        return best_precision, best_recall

    for col in range(ground_truth.shape[1]):
        precision, recall = compute_precision_recall(
            pred, ground_truth.values[0, col], motif_length)
        f_score = compute_f_score(precision, recall)
        if f_score > best_f_score:
            best_precision = precision
            best_recall = recall
            best_f_score = f_score

    return best_precision, best_recall


def as_int_list(values):
    result = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        if isinstance(value, np.ndarray):
            result.extend(as_int_list(value))
        else:
            result.append(int(value))
    return result


def format_seconds(values):
    return [round(float(value), 3) for value in np.asarray(values).reshape(-1)]


def format_dims(dims):
    dims = np.asarray(dims, dtype=object)
    if dims.ndim <= 1:
        if len(dims) == 1 and isinstance(dims[0], np.ndarray):
            return [as_int_list(dims[0])]
        return [as_int_list(dims)]
    return [as_int_list(dim) for dim in dims]


def format_motif_dims(dims, motif_index):
    dims = np.asarray(dims, dtype=object)
    if (
            len(dims) > motif_index
            and isinstance(dims[motif_index], (list, tuple, np.ndarray))
    ):
        return format_dims(dims[motif_index])

    return format_dims(dims)


def benchmark_results_dataframe(results):
    return pd.DataFrame(
        data=results,
        columns=["Dataset", "Method", "Precision", "Recall", "F-Score"])


def print_benchmark_summary(results):
    if len(results) == 0:
        print("No benchmark results to summarize.")
        return

    df = benchmark_results_dataframe(results)
    for column in ["Precision", "Recall", "F-Score"]:
        df[column] = df[column].astype(float)

    summary = (
        df.groupby("Method", sort=False)[["Precision", "Recall", "F-Score"]]
        .mean()
        .round(3)
        .reset_index()
    )
    method_width = max(len("Method"), summary["Method"].str.len().max()) + 2

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Method':<{method_width}} {'Precision':>9} {'Recall':>7} {'F-Score':>8}")
    for _, row in summary.iterrows():
        print(
            f"{row['Method']:<{method_width}} "
            f"{row['Precision']:>9.3f} "
            f"{row['Recall']:>7.3f} "
            f"{row['F-Score']:>8.3f}")


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
        **test_kwargs,
      ):

    motifs_list = []
    dims_list = []

    def _run_method(method_name, callback, *args, **kwargs):
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Method: {method_name}")
        print("=" * 80)
        motif, dims = callback(*args, **kwargs)
        motifs_list.append(motif)
        dims_list.append(dims)

    if "LAMA" in method_names:
        _run_method(
            "LAMA", test_lama, dataset_name, plot=plot, **test_kwargs)
    if "LAMA (naive)" in method_names:
        _run_method(
            "LAMA (naive)", test_lama,
            dataset_name, plot=plot, minimize_pairwise_dist=True, **test_kwargs)
    if "mSTAMP+MDL" in method_names:
        _run_method(
            "mSTAMP+MDL", test_mstamp,
            dataset_name, plot=plot, use_mdl=True, **test_kwargs)
    if "mSTAMP" in method_names:
        _run_method(
            "mSTAMP", test_mstamp,
            dataset_name, plot=plot, use_mdl=False, **test_kwargs)
    if "EMD*" in method_names:
        _run_method(
            "EMD*", test_emd_pca, dataset_name, plot=plot, **test_kwargs)
    if "K-Motifs (TOP-f)" in method_names:
        _run_method(
            "K-Motifs (TOP-f)", test_kmotifs,
            dataset_name, first_dims=True, plot=plot, **test_kwargs)
    if "K-Motifs (all)" in method_names:
        _run_method(
            "K-Motifs (all)", test_kmotifs,
            dataset_name, first_dims=False, plot=plot, **test_kwargs)
    if "SMM" in method_names:
        _run_method("SMM", test_smm, dataset_name, plot=plot)

    # Distances
    if "LAMA (cid)" in method_names:
        _run_method(
            "LAMA (cid)", test_lama,
            dataset_name, plot=plot, distance="cid", **test_kwargs)
    if "LAMA (ed)" in method_names:
        _run_method(
            "LAMA (ed)", test_lama,
            dataset_name, plot=plot, distance="ed", **test_kwargs)
    if "LAMA (cosine)" in method_names:
        _run_method(
            "LAMA (cosine)", test_lama,
            dataset_name, plot=plot, distance="cosine", **test_kwargs)

    # Exclusion Zones
    if "LAMA (alpha=0)" in method_names:
        _run_method(
            "LAMA (alpha=0)", test_lama,
            dataset_name, plot=plot, exclusion_range=0.0, **test_kwargs)
    if "LAMA (alpha=0.25)" in method_names:
        _run_method(
            "LAMA (alpha=0.25)", test_lama,
            dataset_name, plot=plot, exclusion_range=0.25, **test_kwargs)
    if "LAMA (alpha=0.5)" in method_names:
        _run_method(
            "LAMA (alpha=0.5)", test_lama,
            dataset_name, plot=plot, exclusion_range=0.50, **test_kwargs)
    if "LAMA (alpha=0.75)" in method_names:
        _run_method(
            "LAMA (alpha=0.75)", test_lama,
            dataset_name, plot=plot, exclusion_range=0.75, **test_kwargs)
    if "LAMA (alpha=1)" in method_names:
        _run_method(
            "LAMA (alpha=1)", test_lama,
            dataset_name, plot=plot, exclusion_range=1.0, **test_kwargs)

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
    out_file = _PROJECT_ROOT / "tests" / "results" / f"{file_prefix}_{dataset_name}.gzip"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, compression='gzip')


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
    results_dir = _PROJECT_ROOT / "tests" / "results"
    results_file = results_dir / f"{file_prefix}_{dataset_name}.gzip"
    df_loc = pd.read_parquet(results_file)

    available_methods = [
        column for column in df_loc.columns
        if column not in ["dataset", "k"] and not column.endswith("_dims")
    ]
    selected_method_names = [
        method for method in method_names
        if method in df_loc.columns and f"{method}_dims" in df_loc.columns
    ]
    missing_methods = [
        method for method in method_names
        if method not in selected_method_names
    ]

    if missing_methods:
        print(f"Skipping methods without saved results: {missing_methods}")
        print(f"Available methods: {available_methods}")

    if len(selected_method_names) == 0:
        raise ValueError(
            f"{results_file} does not contain any requested methods. "
            f"Requested methods: {method_names}. "
            f"Available methods: {available_methods}.")

    motifs = []
    dims = []
    for id in range(df_loc.shape[0]):
        for motif_method in selected_method_names:
            motifs.append(df_loc.loc[id][motif_method])
            dims.append(df_loc.loc[id][motif_method + "_dims"])

    # write results to file
    for id in range(df_loc.shape[0]):
        for method, motif_set in zip(
                selected_method_names,
                motifs[id * len(selected_method_names): (id + 1) * len(selected_method_names)]
        ):
            precision, recall = compute_best_precision_recall(
                np.sort(motif_set), ground_truth, motif_length)
            f_score = compute_f_score(precision, recall)
            results.append([ds_name, method, precision, recall, f_score])

    if plot:
        for plot_name in all_plot_names:
            plot_names = [
                name for name in all_plot_names[plot_name]
                if name in selected_method_names
            ]
            if len(plot_names) == 0:
                continue

            positions = [selected_method_names.index(name) for name in plot_names]
            out_path = results_dir / "images" / f"{dataset_name}{plot_name}.pdf"
            out_path.parent.mkdir(parents=True, exist_ok=True)
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
