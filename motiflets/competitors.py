import stumpy

from motiflets.plotting import *


def run_mstamp(df, ds_name, motif_length):
    series = df.values.astype(np.float64)

    # Find the Pair Motif
    mps, indices = stumpy.mstump(series, m=motif_length)
    motifs_idx = np.argmin(mps, axis=1)
    nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]

    # Find the optimal dimensionality by minimizing the MDL
    mdls, subspaces = stumpy.mdl(series, motif_length, motifs_idx, nn_idx)
    k = np.argmin(mdls)

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

    fig, ax = plot_motifsets(
        ds_name,
        df,
        motifsets=motifs,
        motiflet_dims=dims,
        motifset_names=motifset_names,
        motif_length=motif_length,
        show=True)
