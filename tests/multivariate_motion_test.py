from matplotlib.animation import FuncAnimation

import amc.amc_parser as amc_parser
from leitmotifs.plotting import *
from leitmotifs.lama import read_ground_truth
import matplotlib as mpl
from leitmotifs.competitors import *

mpl.rcParams['figure.dpi'] = 150


def get_joint_pos_dict(c_joints, c_motion):
    c_joints['root'].set_motion(c_motion)
    out_dict = {}
    for k1, v1 in c_joints['root'].to_dict().items():
        for k2, v2 in zip('xyz', v1.coordinate[:, 0]):
            out_dict['{}_{}'.format(k1, k2)] = v2
    return out_dict


def exclude_body_joints(df):
    # Filter body joints as suggested by Yeh
    exclude = ['root', 'lowerback', 'upperback',
               'thorax', 'lowerneck', 'upperneck', 'head']
    exclude_bones = []
    exclude_bones.extend([x + "_" + k for x in exclude for k in 'xyz'])
    exclude_bones

    return df[~df.index.isin(exclude_bones)]


def include_joints(df, include, add_xyz=True):
    include_bones = []

    if add_xyz:
        include_bones.extend([x + "_" + k for x in include for k in 'xyz'])
    else:
        include_bones = include

    return df[df.index.isin(include_bones)]


def read_motion_dataset(add_xyz=True):
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)
    ground_truth = read_ground_truth(amc_path)
    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = include_joints(exclude_body_joints(df), used_joints, add_xyz=add_xyz)

    # 120 FPS
    time = np.arange(0, df.shape[1] / 120, 1 / 120)
    df.columns = time[:len(df.columns)]
    df.name = ds_name

    return df, ground_truth, joints, motions


def draw_frame(ax, motions, joints, i, joints_to_highlight=None):
    ax.cla()
    ax.grid(False)
    plt.grid(visible=None)
    ax.set_axis_off()

    # Wide
    ax.set_xlim3d(-50, 30)
    ax.set_ylim3d(-20, 40)

    # Limited
    # ax.set_xlim3d(-20, 10)
    # ax.set_ylim3d(-20, 10)

    # ax.set_zlim3d(-20, 40)

    joints['root'].set_motion(motions[i])

    c_joints = joints['root'].to_dict()
    for joint in c_joints.values():
        xs = (joint.coordinate[0, 0])
        ys = (joint.coordinate[1, 0])
        zs = (joint.coordinate[2, 0])
        color = 'r.' if joint.name in joints_to_highlight else 'b.'

        ax.plot(zs, xs, ys, color)

    for joint in c_joints.values():
        child = joint
        if child.parent is not None:
            parent = child.parent
            xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
            ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
            zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]

            color = 'r' if (child.name in joints_to_highlight) and (
                    parent.name in joints_to_highlight) else 'b'
            ax.plot(zs, xs, ys, color)


# http://mocap.cs.cmu.edu/search.php?subjectnumber=13&motion=%

datasets = {
    "Swordplay": {
        "ks": [6],
        "n_dims": 8,
        "motif_length": 120,
        "amc_name": "02_07",
        "asf_path": '../datasets/motion_data/02.asf',
        "slack": 1.0,
        "used_joints": [
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
            'rfemur', 'rtibia', 'rfoot', 'rtoes'
        ]
    },
    "Basketball": {
        "ks": [5],
        "n_dims": 5,
        "motif_length": 50,
        "amc_name": "06_02",
        "asf_path": '../datasets/motion_data/06.asf',
        "slack": 1.0,
        "used_joints": [
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
            'rfemur', 'rtibia', 'rfoot', 'rtoes'
        ]
    },
    "Boxing": {
        "ks": [10],
        "motif_length": 130,
        "amc_name": "13_17",
        "n_dims": 7,
        "slack": 0.5,
        "asf_path": '../datasets/motion_data/13.asf',
        "used_joints": [
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
            'rfemur', 'rtibia', 'rfoot', 'rtoes'
        ]
    },
    "Charleston - Side By Side Female": {
        "ks": [3],
        "motif_length": 140,
        "n_dims": 10,
        "amc_name": "93_04",
        "slack": 0.5,
        "asf_path": '../datasets/motion_data/93.asf',
        "used_joints": [
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand',
            'rfingers', 'rthumb', 'rfemur', 'rtibia', 'rfoot', 'rtoes',
            # 'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand',
            # 'lfingers', 'lthumb', 'lfemur', 'ltibia', 'lfoot', 'ltoes'
        ]
    },
    # "Charleston-Fancy": {
    #     "ks": [3],
    #     "motif_length": 120,
    #     "n_dims": 10,
    #     "slack": 0.5,
    #     "amc_name": "93_08",  # Fancy Charleston
    #     "asf_path": '../datasets/motion_data/93.asf'
    # },
    # "Charleston-Side-By-Side-Male": {
    #     "ks": [3],
    #     "motif_length": 140,
    #     "n_dims": 10,
    #     "amc_name": "93_05",
    #     "slack": 0.5,
    #     "asf_path": '../datasets/motion_data/93.asf',
    #     "used_joints": [
    #         # 'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand',
    #         # 'rfingers', 'rthumb', 'rfemur', 'rtibia', 'rfoot', 'rtoes',
    #         'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand',
    #         'lfingers', 'lthumb', 'lfemur', 'ltibia', 'lfoot', 'ltoes'
    #     ]
    # }
}

# Switch to write videos to disc
write_video = False


def get_ds_parameters(name):
    global ds_name, dataset, ks, used_joints, n_dims, slack
    global motif_length, amc_name, asf_path, amc_path, k_max

    ds_name = name
    dataset = datasets[ds_name]
    ks = dataset["ks"]
    used_joints = dataset["used_joints"]
    n_dims = dataset["n_dims"]
    motif_length = dataset["motif_length"]
    slack = dataset["slack"]
    amc_name = dataset["amc_name"]
    asf_path = dataset["asf_path"]
    amc_path = '../datasets/motion_data/' + amc_name + '.amc'

    # for learning parameters
    k_max = np.max(dataset["ks"]) + 2
    m = dataset["motif_length"]


# All joints
# use_joints = np.asarray(
#    ['root', 'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head',
#     'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
#     'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
#     'rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes'])
#
# Body
# used_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']
#
# Right
# used_joints = ['rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb']
#
# Right Body
# used_joints = [
#    'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
#    'rfemur', 'rtibia', 'rfoot', 'rtoes'
# ]
#
# Hands
# used_joints = [
#    'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
#    'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb']
# footwork
# used_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']


def test_lengths():
    dataset_names = [
        "Boxing",
        "Swordplay",
        "Basketball",
        "Charleston - Side By Side Female"
    ]

    for ds in dataset_names:
        get_ds_parameters(ds)
        df, ground_truth, joints, motions = read_motion_dataset()
        print(ds, df.shape)


def test_ground_truth():
    get_ds_parameters("Swordplay")
    df, ground_truth, joints, motions = read_motion_dataset()

    pos = np.array([1.5, 3.5, 5.5, 7.7, 9.6, 16.9])
    length = 120 / 120

    positions = []
    for p in pos:
        positions.append(np.int32([p * 120, (p + length) * 120]))
    positions = np.array([positions], dtype=np.int32)
    print(positions)

    motif = positions[0][:, 0]
    filtered_joints = list(set([joint[:-2] for joint in used_joints]))
    for j, pos in enumerate(motif):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        out_path = 'video/anim/motif_' + amc_name + '_' + str(j) + '.gif'

        FuncAnimation(fig,
                      lambda i: draw_frame(ax, motions, joints, i,
                                           joints_to_highlight=filtered_joints),
                      range(pos, pos + motif_length, 4)).save(
            out_path,
            bitrate=100,
            fps=20)

    ground_truth = read_ground_truth(amc_path)
    ml = LAMA(ds_name, df,
              dimension_labels=df.index,
              ground_truth=ground_truth
              )
    ml.plot_dataset()


#"Boxing",
#"Swordplay",
#"Basketball",
#"Charleston - Side By Side Female"

def test_lama(
        dataset_name="Charleston - Side By Side Female",
        minimize_pairwise_dist=False,
        use_PCA=False,
        motifset_name="LAMA",
        plot=True):
    get_ds_parameters(dataset_name)
    df, ground_truth, joints, motions = read_motion_dataset()

    # make the signal uni-variate by applying PCA
    if use_PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        df_transform = pca.fit_transform(df.T).T
    else:
        df_transform = df

    # m = 120
    # ks = 6
    # k_max = ks + 2
    # n_dims = 10

    # for use_dims in np.arange(2, n_dims):
    ml = LAMA(
        amc_name, df_transform,
        dimension_labels=df.index,
        n_dims=n_dims,
        slack=slack,
        ground_truth=ground_truth,
        minimize_pairwise_dist=minimize_pairwise_dist
    )

    # learn parameters
    # length_range = np.arange(50, 200, 10)
    # print(length_range)
    # m, all_minima = ml.fit_motif_length(
    #     k_max, length_range,
    #     plot=True,
    #     plot_best_only=True,
    #     plot_motifsets=True)

    dists, motif_sets, elbow_points = ml.fit_k_elbow(
        k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifsets=False)

    print("Positions (Frame):", np.sort(motif_sets[ks]))
    print("Time:", np.sort(motif_sets[ks]) / 120)  # 120 FPS
    # ml.plot_motifset(elbow_points=ks, motifset_name=motifset_name)

    if use_PCA:
        dims = np.argsort(pca.components_[:])[:, :n_dims]
    else:
        dims = ml.leitmotifs_dims[ks]

    for a, i in enumerate(ks):
        motif = motif_sets[i]

        print("Positions (Frame):", np.sort(motif))
        print("Time:", np.sort(motif) / 120)  # 120 FPS

        if use_PCA:
            print("\tdims\t:", repr(dims))
        else:
            print("\tdims\t:", repr(dims[a]))

        if plot:
            ml.plot_motifset(elbow_points=[i], motifset_name=motifset_name)

        if write_video:
            generate_gif(motif, motif_length, motions, joints)

    # for minimum in all_minima:
    #     m = length_range[minimum]
    #     dists = ml.all_dists[minimum]
    #     elbow_points = ml.all_elbows[minimum]
    #
    #     leitmotifs = ml.all_top_motiflets[minimum]
    #     leitmotifs = np.zeros(len(dists), dtype=object)
    #     leitmotifs[elbow_points] = ml.all_top_motiflets[minimum]
    #
    #     dimensions = np.zeros(len(dists), dtype=object)
    #     dimensions[elbow_points] = ml.all_dimensions[minimum]  # need to unpack
    #     if len(elbow_points) > 1:
    #         for eb in elbow_points:
    #             motif = leitmotifs[eb]
    #             generate_gif(motif, m, motions, joints)

    return motif_sets[ks], dims


def test_emd_pca(dataset_name="Boxing", plot=True):
    return test_lama(dataset_name, use_PCA=True, motifset_name="PCA", plot=plot)


def test_mstamp(dataset_name="Boxing", plot=True, use_mdl=True):
    get_ds_parameters(dataset_name)
    df, ground_truth, joints, motions = read_motion_dataset()
    return run_mstamp(df, ds_name, motif_length=motif_length,
                      ground_truth=ground_truth, plot=plot,
                      use_mdl=use_mdl, use_dims=n_dims)


def test_kmotifs(dataset_name="Boxing", first_dims=True, plot=True):
    get_ds_parameters(dataset_name)
    df, ground_truth, joints, motions = read_motion_dataset()

    motif_sets = []
    used_dims = []
    for target_k in ks:
        motif, dims = run_kmotifs(
            df,
            ds_name,
            motif_length=motif_length,
            slack=slack,
            r_ranges=np.arange(10, 300, 1),
            use_dims=n_dims if first_dims else df.shape[0],  # first dims or all dims
            target_k=target_k,
            ground_truth=ground_truth,
            plot=plot
        )
        used_dims.append(np.arange(dims))
        motif_sets.append(motif)

        if write_video:
            generate_gif(motif, motif_length, motions, joints)

    return motif_sets, used_dims


def generate_gif(motif, m, motions, joints):
    filtered_joints = list(set([joint[:-2] for joint in used_joints]))
    for j, pos in enumerate(motif):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        out_path = 'video/anim/motif_' + amc_name + "_" + str(j) + '.gif'
        FuncAnimation(
            fig,
            lambda i: draw_frame(ax, motions, joints, i,
                                 joints_to_highlight=filtered_joints),
            range(pos, pos + m, 4)).save(
            out_path,
            bitrate=100,
            fps=20)


def test_publication():
    dataset_names = [
        "Boxing",
        "Swordplay",
        "Basketball",
        "Charleston - Side By Side Female"
    ]
    method_names = [
        "LAMA",
        "LAMA (naive)",
        "mSTAMP+MDL",
        "mSTAMP",
        "EMD*",
        "K-Motifs (TOP-f)",
        "K-Motifs (all)"
    ]
    plot = False
    for dataset_name in dataset_names:
        motifA, dimsA = test_lama(dataset_name, plot=plot)
        motifB, dimsB = test_lama(dataset_name, plot=plot, minimize_pairwise_dist=True)
        motifC, dimsC = test_mstamp(dataset_name, plot=plot, use_mdl=True)
        motifD, dimsD = test_mstamp(dataset_name, plot=plot, use_mdl=False)
        motifE, dimsE = test_emd_pca(dataset_name, plot=plot)
        motifF, dimsF = test_kmotifs(dataset_name, first_dims=True, plot=plot)
        motifG, dimsG = test_kmotifs(dataset_name, first_dims=False, plot=plot)

        method_names_dims = [name+"_dims" for name in method_names]
        columns = ["dataset", "k"]
        columns.extend(method_names)
        columns.extend(method_names_dims)
        df = pd.DataFrame(columns=columns)

        for i, k in enumerate(ks):
            df.loc[len(df.index)] \
                = [dataset_name, k,
                   motifA[i].tolist(), motifB[i].tolist(), motifC[0], motifD[0],
                   motifE[i].tolist(), motifF[i].tolist(), motifG[i].tolist(),
                   dimsA[i].tolist(), dimsB[i].tolist(), dimsC[0].tolist(), dimsD[0].tolist(),
                   dimsE[i].tolist(), dimsF[i].tolist(), dimsG[i].tolist()]

        print("--------------------------")

        # from datetime import datetime
        # currentDateTime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        df.to_parquet(
            f'results/results_motion_{dataset_name}.gzip',  # _{currentDateTime}
            compression='gzip')


def test_plot_results():
    dataset_names = [
        "Boxing",
        #"Swordplay",
        #"Basketball",
        #"Charleston - Side By Side Female"
    ]

    method_names = [
        "LAMA",
        "LAMA (naive)",
        "mSTAMP+MDL",
        "mSTAMP",
        "EMD*",
        "K-Motifs (TOP-f)",
        "K-Motifs (all)"
    ]
    results = []

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        df, ground_truth, joints, motions = read_motion_dataset()

        df_loc = pd.read_parquet(
            f"results/results_motion_{dataset_name}.gzip")

        motifs = []
        dims = []
        for id in range(df_loc.shape[0]):
            for motif_method in method_names:
                motifs.append(df_loc.loc[id][motif_method])
                dims.append(df_loc.loc[id][motif_method+"_dims"])

        # write results to file
        for id in range(df_loc.shape[0]):
            for method, motif_set in zip(
                    method_names,
                    motifs[id * len(method_names): (id + 1) * len(method_names)]
            ):
                precision, recall = compute_precision_recall(
                    np.sort(motif_set), ground_truth.values[0, 0], motif_length)
                results.append([ds_name, method, precision, recall])

        pd.DataFrame(
            data=np.array(results),
            columns=["Dataset", "Method", "Precision", "Recall"]).to_csv(
            "results/motion_precision.csv")

        print(results)

        if True:
            plot_names = [
                "mSTAMP+MDL",
                "mSTAMP",
                "EMD*",
                # "K-Motifs (TOP-f)",
                "K-Motifs (all)",
                "LAMA",
            ]

            # FIXME this will not work with multiple motifsets
            positions = [method_names.index(name) for name in plot_names]
            out_path = "results/images/" + dataset_name + "_new.pdf"
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



def test_motif_length():
    dataset_names = [
        "Boxing",
        "Swordplay",
        "Basketball",
        "Charleston - Side By Side Female"
    ]

    results = []

    for dataset_name in dataset_names:
        get_ds_parameters(dataset_name)
        df, ground_truth, joints, motions = read_motion_dataset()

        print(dataset_name, np.diff(ground_truth.iloc[0, 0], axis=1) / 120)