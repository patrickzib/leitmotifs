import time
from pathlib import Path
from unittest.mock import patch
import warnings

from leitmotifs.plotting import *
from leitmotifs.lama import (
    _sliding_dot_product, _update_sliding_dot_product, _argknn,
    get_pairwise_extent_raw, run_LAMA, find_au_ef_motif_length,
    search_leitmotifs_elbow)
from leitmotifs.distances import (
    complexity_invariant_distance, complexity_invariant_distance_single,
    cosine_distance, cosine_distance_single, sliding_csum, sliding_csum_dcsum,
    znormed_euclidean_distance_single)
from leitmotifs.competitors import (
    benchmark_results_dataframe, compute_best_precision_recall,
    compute_f_score, eval_tests, format_motif_dims)
from numba import get_num_threads, prange

warnings.simplefilter("ignore")

path = "./datasets/experiments/"


def test_lama_imports_are_backwards_compatible():
    from leitmotifs import LAMA as package_lama
    from leitmotifs.lama import LAMA as module_lama
    from leitmotifs.plotting import LAMA as plotting_lama

    assert package_lama is module_lama
    assert plotting_lama is module_lama


def test_compute_f_score():
    assert compute_f_score(1.0, 1.0) == 1.0
    assert compute_f_score(0.0, 0.0) == 0.0
    np.testing.assert_allclose(compute_f_score(0.5, 1.0), 2 / 3)


def test_compute_best_precision_recall_uses_best_ground_truth_column():
    ground_truth = pd.DataFrame({
        "B": [np.array([[0, 5], [10, 15]])],
        "C": [np.array([[100, 105], [120, 125]])],
    })

    precision, recall = compute_best_precision_recall(
        np.array([100, 120]), ground_truth, 5)

    assert precision == 1.0
    assert recall == 1.0


def test_benchmark_results_dataframe_includes_f_score():
    df = benchmark_results_dataframe([
        ["Dataset A", "LAMA", 0.5, 1.0, compute_f_score(0.5, 1.0)]
    ])

    assert df.columns.tolist() == [
        "Dataset", "Method", "Precision", "Recall", "F-Score"]
    np.testing.assert_allclose(df.loc[0, "F-Score"], 2 / 3)


def test_format_motif_dims_selects_matching_motif():
    dims = np.array([np.array([0, 2]), np.array([1, 3])], dtype=object)

    assert format_motif_dims(dims, 0) == [[0, 2]]
    assert format_motif_dims(dims, 1) == [[1, 3]]


def test_cosine_distance_matches_direct_formula():
    ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0])
    m = 3
    n = len(ts) - m + 1
    order = 0
    subsequences = np.array([ts[i:i + m] for i in range(n)])
    dot_rolled = np.array([
        np.dot(subsequences[order], subsequence)
        for subsequence in subsequences
    ])
    preprocessing = sliding_csum(ts, m)

    actual = cosine_distance(
        dot_rolled.copy(), n, m, preprocessing, order, halve_m=0)
    expected = np.array([
        1 - np.dot(subsequences[order], subsequence) / (
            np.linalg.norm(subsequences[order]) * np.linalg.norm(subsequence))
        for subsequence in subsequences
    ])
    expected[order] = 0.0

    np.testing.assert_allclose(actual, expected)
    for i, expected_distance in enumerate(expected):
        actual_distance = cosine_distance_single(
            subsequences[order], subsequences[i], order, i, preprocessing)
        np.testing.assert_allclose(actual_distance, expected_distance)


def test_complexity_invariant_distance_matches_direct_formula():
    ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0])
    m = 3
    n = len(ts) - m + 1
    order = 0
    subsequences = np.array([ts[i:i + m] for i in range(n)])
    dot_rolled = np.array([
        np.dot(subsequences[order], subsequence)
        for subsequence in subsequences
    ])
    preprocessing = sliding_csum_dcsum(ts, m)

    direct_complexity = np.array([
        np.sqrt(np.sum(np.diff(subsequence) ** 2))
        for subsequence in subsequences
    ])
    np.testing.assert_allclose(preprocessing[1], direct_complexity)

    def direct_cid(a, b):
        ed = np.sum((a - b) ** 2)
        ce_a = np.sqrt(np.sum(np.diff(a) ** 2))
        ce_b = np.sqrt(np.sum(np.diff(b) ** 2))
        cf = max(max(ce_a, ce_b) / max(min(ce_a, ce_b), 1e-12), 1.0)
        return ed * cf

    actual = complexity_invariant_distance(
        dot_rolled.copy(), n, m, preprocessing, order, halve_m=0)
    expected = np.array([
        direct_cid(subsequences[order], subsequence)
        for subsequence in subsequences
    ])
    expected[order] = 0.0

    np.testing.assert_allclose(actual, expected)
    for i, expected_distance in enumerate(expected):
        actual_distance = complexity_invariant_distance_single(
            subsequences[order], subsequences[i], order, i, preprocessing)
        np.testing.assert_allclose(actual_distance, expected_distance)


def test_eval_tests_skips_missing_methods(capsys):
    dataset_name = "tmp_eval_missing_methods"
    results_file = Path("tests/results") / f"tmp_{dataset_name}.gzip"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dataset": [dataset_name],
        "k": [2],
        "LAMA": [np.array([0, 10])],
        "LAMA_dims": [np.array([0])],
    }).to_parquet(results_file, compression="gzip")

    ground_truth = pd.Series([np.array([[0, 5], [10, 15]])]).to_frame().T
    results = []
    try:
        eval_tests(
            dataset_name=dataset_name,
            ds_name=dataset_name,
            df=pd.DataFrame(np.zeros((1, 20))),
            method_names=["LAMA", "LAMA (naive)"],
            motif_length=5,
            ground_truth=ground_truth,
            all_plot_names={"_new": ["LAMA", "LAMA (naive)"]},
            file_prefix="tmp",
            results=results,
            plot=False,
        )
    finally:
        results_file.unlink()

    assert [row[1] for row in results] == ["LAMA"]
    assert "Skipping methods without saved results: ['LAMA (naive)']" in capsys.readouterr().out


def test_convert_to_2d_raises_value_error_for_transposed_input():
    with np.testing.assert_raises(ValueError):
        convert_to_2d(np.zeros((5, 2)))


def test_fit_k_elbow_requires_motif_length():
    ml = LAMA(
        "Missing motif length",
        np.zeros((1, 40), dtype=np.float64),
        n_jobs=1,
    )

    with np.testing.assert_raises(ValueError):
        ml.fit_k_elbow(3, plot_elbows=False, plot_motifsets=False)


def test_fit_motif_length_rejects_invalid_ranges():
    data = np.zeros((1, 40), dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        find_au_ef_motif_length(data, 3, np.array([], dtype=np.int32), n_jobs=1)

    ml = LAMA("Invalid motif lengths", data, n_jobs=1)
    with np.testing.assert_raises(ValueError):
        ml.fit_motif_length(
            3, np.array([0, 100], dtype=np.int32),
            plot=False, plot_elbows=False, plot_motifsets=False)


def test_fit_dimensions_uses_configured_backend():
    backends = []

    def fake_search(*args, **kwargs):
        backends.append(kwargs["backend"])
        return (
            np.array([0.0, 0.0, 1.0, 2.0]),
            np.array([None, None, np.array([0, 10]), np.array([0, 10, 20])],
                     dtype=object),
            np.array([None, None, np.array([0]), np.array([0])], dtype=object),
            np.array([2], dtype=np.int32),
            None,
            None,
            0.0,
        )

    with patch("leitmotifs.lama.search_leitmotifs_elbow", side_effect=fake_search), \
            patch("leitmotifs.lama._plotting") as plotting:
        plotting.return_value.plt.subplots.return_value = (
            object(), type("Axes", (), {"set_title": lambda self, title: None})())
        ml = LAMA(
            "Backend passthrough",
            np.zeros((2, 40), dtype=np.float64),
            backend="scalable",
            n_jobs=1,
        )
        ml.fit_dimensions(3, 8, np.array([1, 2], dtype=np.int32))

    assert backends == ["scalable", "scalable"]


def test_search_leitmotifs_elbow_rejects_invalid_motif_length():
    data = np.zeros((1, 40), dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        search_leitmotifs_elbow(3, data, 0, n_jobs=1)

    with np.testing.assert_raises(ValueError):
        search_leitmotifs_elbow(3, data, 40, n_jobs=1)


def test_search_leitmotifs_elbow_restores_numba_threads_on_error():
    data = np.zeros((1, 40), dtype=np.float64)
    previous_threads = get_num_threads()

    with np.testing.assert_raises(ValueError):
        search_leitmotifs_elbow(3, data, 8, n_jobs=1, backend="invalid")

    assert get_num_threads() == previous_threads


def test_update_sliding_dot_product_matches_roll_update():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=40).astype(np.float64)
    m = 7
    n = ts.shape[0] - m + 1
    dot_first = _sliding_dot_product(ts[:m], ts)
    dot_old = _sliding_dot_product(ts[3:3 + m], ts)
    dot_new = dot_old.copy()

    for order in range(4, 12):
        dot_old = (
                np.roll(dot_old, 1)
                + ts[order + m - 1] * ts[m - 1:n + m]
                - ts[order - 1] * np.roll(ts[:n], 1)
        )
        dot_old[0] = dot_first[order]
        _update_sliding_dot_product(dot_new, dot_first[order], ts, order, m, n)
        np.testing.assert_allclose(dot_new, dot_old)


def test_run_lama_returns_dims_used_for_extent():
    ts = np.arange(8, dtype=np.float64).reshape(2, 4)
    D = np.zeros((2, 4, 4), dtype=np.float64)
    D[0, 0, 2] = D[0, 2, 0] = 100.0
    D[1, 0, 2] = D[1, 2, 0] = 1.0

    knns = np.full((2, 4, 2), -1, dtype=np.int32)
    knns[0, 0] = np.array([0, 2], dtype=np.int32)
    dim_index = np.array([[0], [0], [1], [0]], dtype=np.int32)
    preprocessing = np.array(
        [sliding_mean_std(ts[d], 1) for d in range(ts.shape[0])],
        dtype=np.float64)

    candidate, dist, dims = run_LAMA(
        ts, 1, 2, D, knns, dim_index,
        distance_single=znormed_euclidean_distance_single,
        preprocessing=preprocessing,
        use_D_full=True)

    np.testing.assert_array_equal(candidate, np.array([0, 2], dtype=np.int32))
    assert dist == 1.0
    np.testing.assert_array_equal(dims, np.array([1], dtype=np.int32))


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series

def test_penguins_multivariate():
    lengths = [1_000,
               5_000,
               10_000,
               # 30_000,
               # 50_000,
               # 100_000,
               # 150_000, 200_000,
               # 250_000
               ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length, flush=True)
        series = B.iloc[:length].T

        ml = LAMA(
            ds_name,
            series,
            n_dims=3,
            backend="scalable",
            n_jobs=-1,
        )

        k_max = 5

        t_before = time.time()
        dists, motif, elbow_points = ml.fit_k_elbow(
            k_max,
            motif_length=22,
            plot_elbows=False,
            plot_motifsets=False
        )
        t_after = time.time()
        time_s[i] = t_after - t_before
        memory_usage = ml.memory_usage

        print("\tTime:", time_s[i])
        print("\tMemory:", memory_usage, "MB")


        start = motif[-1]
        dims = ml.leitmotifs_dims[-1]
        best_motiflet, min_extent = compute_knn(
                series.to_numpy(),
                motif=start,
                m=22,
                dims=dims,
                k=k_max)

        print("\tBest motiflet:", best_motiflet)
        print("\tMinimum extent:", min_extent)


# @njit(cache=True)
def compute_knn(
        ts,
        motif,
        m,
        dims,
        k,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        slack=0.5
):
    halve_m = np.int32(m * slack)
    n = ts.shape[-1] - m + 1

    preprocessing = []
    for d in dims:
        preprocessing.append(distance_preprocessing(ts[d], m))
    preprocessing = np.array(preprocessing, dtype=np.float64)

    knns = np.zeros((len(motif), k), dtype=np.int32)
    extents = np.zeros(len(motif), dtype=np.float64)

    for i in range(len(motif)):
        dist = 0.0
        start = motif[i]
        for j, d in enumerate(dims):
            dot_rolled = _sliding_dot_product(ts[d, start : start + m], ts[d])
            dist += distance(dot_rolled, n, m, preprocessing[j], start, halve_m)

        knn = _argknn(dist, k, m, slack=slack)
        knns[i] = knn
        extents[i] = get_pairwise_extent_raw(
            ts[dims], knns[i], np.arange(len(dims), dtype=np.int32),
            m, distance_single, preprocessing)

    min_pos = np.argmin(extents)
    best_motiflet = knns[min_pos]
    min_extent = extents[min_pos]

    return best_motiflet, min_extent
