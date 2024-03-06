import stumpy

from motiflets.plotting import *
from numba import njit


def run_mstamp(df, ds_name, motif_length, ground_truth=None, plot=True):
    series = df.values.astype(np.float64)

    # Find the Pair Motif
    mps, indices = stumpy.mstump(series, m=motif_length)
    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    # Find the optimal dimensionality by minimizing the MDL
    mdls, subspaces = stumpy.mdl(series, motif_length, motifs_idx, nn_idx)
    k = np.argmin(mdls)

    if plot:
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

    dims = [subspaces[k]]
    motifs = [[motifs_idx[subspaces[k]][0], nn_idx[subspaces[k]][0]]]
    motifset_names = ["mStamp"]

    if plot:
        _ = plot_motifsets(
            ds_name,
            df,
            motifsets=motifs,
            motiflet_dims=dims,
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
            print(f"cardinality: {cardinality} for r={r}")
            last_cardinality = cardinality

        if cardinality >= target_k:
            print(f"Radius: {r}, K: {cardinality}")
            print(f"Pos: {motifset}")
            motifset_names = ["K-Motif"]

            if plot:
                plot_motifsets(
                    ds_name,
                    series,
                    motifsets=motifset.reshape(1, -1),
                    motiflet_dims=np.arange(use_dims).reshape(1, -1),
                    motifset_names=motifset_names,
                    motif_length=motif_length,
                    ground_truth=ground_truth,
                    show=True)

            return motifset, use_dims

    return [], []
