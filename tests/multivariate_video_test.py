from matplotlib.animation import FuncAnimation

import amc.amc_parser as amc_parser
from motiflets.plotting import *
from motiflets.motiflets import read_ground_truth
import matplotlib as mpl
from motiflets.competitors import *

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


# plot multi-variat motiflet
def plot_multivariate_motiflet(
        data, motifset, m, d=[], names=[]
):
    fig, axes = plt.subplots(len(data) + 1, 1, figsize=(14, 2 * len(data)))

    for i in range(len(data)):
        ax = axes[i]
        # plt.subplot(len(data) + 1, 1, i + 1)
        if len(names) == len(data):
            ax.set_title('${{{0}}}$'.format(names[i]))
        else:
            ax.set_title('$T_{{{0}}}$'.format(i + 1))

        for idx, pos in enumerate(motifset):
            ax.plot(range(0, m), data[i, :][pos:pos + m])  # c=color[idx])

        ax.set_xlim((0, m))

    plt.tight_layout()
    plt.show()


# http://mocap.cs.cmu.edu/search.php?subjectnumber=13&motion=%

datasets = {
    "Swordplay": {
        "ks": [6],
        "n_dims": 10,
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
        "n_dims": 10,
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
        "motif_length": 100,
        "amc_name": "13_17",
        "n_dims": 7,
        "slack": 0.5,
        "asf_path": '../datasets/motion_data/13.asf',
        "used_joints": [
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
            'rfemur', 'rtibia', 'rfoot', 'rtoes'
        ]
    },
    "Charleston-Fancy": {
        "ks": [3],
        "motif_length": 120,
        "n_dims": 10,
        "slack": 0.5,
        "amc_name": "93_08",  # Fancy Charleston
        "asf_path": '../datasets/motion_data/93.asf'
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
    "Charleston-Side-By-Side-Male": {
        "ks": [3],
        "motif_length": 140,
        "n_dims": 10,
        "amc_name": "93_05",
        "slack": 0.5,
        "asf_path": '../datasets/motion_data/93.asf',
        "used_joints": [
            # 'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand',
            # 'rfingers', 'rthumb', 'rfemur', 'rtibia', 'rfoot', 'rtoes',
            'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand',
            'lfingers', 'lthumb', 'lfemur', 'ltibia', 'lfoot', 'ltoes'
        ]
    }
}


def get_ds_parameters(name):
    global ds_name, dataset, ks, m, used_joints, n_dims, slack
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

    filtered_joints = list(set([joint[:-2] for joint in used_joints]))
    for j, pos in enumerate(positions[0][:, 0]):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        out_path = 'video/anim/motiflet_' + amc_name + '_' + str(j) + '.gif'

        FuncAnimation(fig,
                      lambda i: draw_frame(ax, motions, joints, i,
                                           joints_to_highlight=filtered_joints),
                      range(pos, pos + motif_length, 4)).save(
            out_path,
            bitrate=100,
            fps=20)

    ground_truth = read_ground_truth(amc_path)
    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   ground_truth=ground_truth
                   )
    ml.plot_dataset()


def test_lama():
    get_ds_parameters("Swordplay")
    df, ground_truth, joints, motions = read_motion_dataset()

    m = 120
    ks = 6
    k_max = ks + 2
    n_dims = 10

    ml = Motiflets(
        amc_name, df,
        dimension_labels=df.index,
        n_dims=n_dims,
        ground_truth=ground_truth,
        slack=slack
    )

    # length_range = np.arange(50, 200, 10)
    # print(length_range)
    # m, all_minima = ml.fit_motif_length(
    #     k_max, length_range,
    #     plot=True,
    #     plot_best_only=True,
    #     plot_motifsets=True)

    dists, candidates, elbow_points = ml.fit_k_elbow(
        k_max,
        motif_length=m,
        plot_elbows=False,
        plot_motifsets=False)

    print("Positions (Frame):", np.sort(candidates[ks]))
    print("Time:", np.sort(candidates[ks]) / 120)  # 120 FPS
    ml.plot_motifset(elbow_points=[ks], motifset_name="LAMA")

    for i in np.arange(6, 7):
        motiflet = candidates[i]
        use_joints = df.index.values[ml.motiflets_dims[i]]
        filtered_joints = list(set([joint[:-2] for joint in use_joints]))

        print("Positions (Frame):", np.sort(motiflet))
        print("Time:", np.sort(motiflet) / 120)  # 120 FPS
        ml.plot_motifset(elbow_points=[i])

        for j, pos in enumerate(motiflet):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            out_path = 'video/anim/motiflet_' + amc_name + '_' + str(i) + "_" + str(
                j) + '.gif'

            FuncAnimation(fig,
                          lambda i: draw_frame(ax, motions, joints, i,
                                               joints_to_highlight=filtered_joints),
                          range(pos, pos + motif_length, 4)).save(
                out_path,
                bitrate=100,
                fps=20)

    # for minimum in all_minima:
    #   motif_length = length_range[minimum]
    #   dists = ml.all_dists[minimum]
    #   elbow_points = ml.all_elbows[minimum]

    #   motiflets = ml.all_top_motiflets[minimum]
    #   motiflets = np.zeros(len(dists), dtype=object)
    #   motiflets[elbow_points] = ml.all_top_motiflets[minimum]

    #   dimensions = np.zeros(len(dists), dtype=object)
    #   dimensions[elbow_points] = ml.all_dimensions[minimum]  # need to unpack
    # if len(elbow_points) > 1:
    #     for eb in elbow_points:
    #         for i, pos in enumerate(motiflets[eb]):
    #             use_joints = df.index.values[dimensions[eb]]  # FIXME!?
    #             # strip the _x, _y, _z from the joint
    #             use_joints = [joint[:-2] for joint in use_joints]
    #             fig = plt.figure()
    #             ax = plt.axes(projection='3d')
    #
    #             out_path = ('video/motiflet_' + amc_name + '_' + str(
    #                 motif_length)
    #                         + '_' + str(eb) + '_' + str(i) + '.gif')
    #
    #             FuncAnimation(fig,
    #                           lambda i: draw_frame(
    #                               ax, motions, joints, i,
    #                               joints_to_highlight=use_joints
    #                           ),
    #                           range(pos, pos + motif_length, 4)).save(
    #                 out_path,
    #                 bitrate=1000,
    #                 fps=20)


def test_motion_capture():
    get_ds_parameters("Stairs")
    df, ground_truth, joints, motions = read_motion_dataset()

    ml = Motiflets(
        df.name, df,
        dimension_labels=df.index,
        n_dims=n_dims,
        slack=slack
    )

    dists, candidates, elbow_points = ml.fit_k_elbow(
        k_max,
        plot_elbows=True,
        motif_length=motif_length)

    print("----")
    print(dists)
    print(elbow_points)
    print(list(candidates[elbow_points]))
    print("----")

    path_ = "video/motiflet_" + amc_name + "_Channels_" + str(
        len(df.index)) + "_Motif.pdf"
    ml.plot_motifset(path=path_, motifset_name="LAMA")

    motiflets = candidates[elbow_points]
    filtered_joints = list(set([joint[:-2] for joint in used_joints]))

    for i, motiflet in enumerate(motiflets):
        for j, pos in enumerate(motiflet):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            out_path = 'video/motiflet_' + amc_name + '_' + str(i) + '_' + str(
                j) + '.gif'

            FuncAnimation(fig,
                          lambda i: draw_frame(ax, motions, joints, i,
                                               joints_to_highlight=filtered_joints),
                          range(pos, pos + motif_length, 4)).save(
                out_path,
                bitrate=1000,
                fps=20)


def test_lama_chaleston():
    get_ds_parameters("Charleston - Side By Side Female")
    df, ground_truth, joints, motions = read_motion_dataset()

    length_range = np.arange(100, 200, 10)
    print(length_range)

    ml = Motiflets(
        "Charleston Side By Side - Female",
        df,
        dimension_labels=df.index,
        n_dims=n_dims,
        ground_truth=ground_truth,
        slack=slack
    )

    m, all_minima = ml.fit_motif_length(
        k_max, length_range,
        plot=False,
        plot_best_only=True,
        plot_motifsets=True)
    ml.plot_motifset(path="images_paper/charleston.pdf", motifset_name="LAMA")

    print("Positions:")
    for eb in ml.elbow_points:
        motiflet = np.sort(ml.motiflets[eb])
        print("\tpos\t:", repr(motiflet))
        print("\tdims\t:", repr(ml.motiflets_dims[eb]))

    # for minimum in all_minima:
    #   motif_length = length_range[minimum]
    #   dists = ml.all_dists[minimum]
    #   elbow_points = ml.all_elbows[minimum]
    #   motiflets = np.zeros(len(dists), dtype=object)
    #   motiflets[elbow_points] = ml.all_top_motiflets[minimum]
    #   dimensions = np.zeros(len(dists), dtype=object)
    #   dimensions[elbow_points] = ml.all_dimensions[minimum]  # need to unpack

    if True:
        motif_length = ml.motif_length
        elbow_points = ml.elbow_points
        motiflets = ml.motiflets
        dimensions = ml.motiflets_dims

        if len(elbow_points) >= 1:
            for eb in elbow_points:
                for i, pos in enumerate(motiflets[eb]):
                    use_joints = df.index.values[dimensions[eb]]
                    # strip the _x, _y, _z from the joint
                    use_joints = [joint[:-2] for joint in use_joints]
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')

                    out_path = ('images_paper/charleston_'
                                + amc_name + '_' + str(motif_length)
                                + '_' + str(eb) + '_' + str(i) + '.gif')

                    FuncAnimation(fig,
                                  lambda i: draw_frame(
                                      ax, motions, joints, i,
                                      joints_to_highlight=use_joints
                                  ),
                                  range(pos, pos + motif_length, 4)).save(
                        out_path,
                        bitrate=1000,
                        fps=20)

                    # break


def test_mstamp():
    get_ds_parameters("Basketball")
    df, ground_truth, joints, motions = read_motion_dataset()
    run_mstamp(df, ds_name, motif_length=m, ground_truth=ground_truth)


def test_kmotifs():
    # get_ds_parameters("Charleston-Side-By-Side-Male")
    # get_ds_parameters("Basketball")
    get_ds_parameters("Swordplay")
    df, ground_truth, joints, motions = read_motion_dataset()

    for target_k in ks:
        motif = run_kmotifs(
            df,
            ds_name,
            m,
            slack=slack,
            r_ranges=np.arange(10, 300, 1),
            use_dims=n_dims,
            target_k=target_k,
            ground_truth=ground_truth
        )

        filtered_joints = list(set([joint[:-2] for joint in used_joints]))
        for j, pos in enumerate(motif[-1]):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            out_path = 'video/anim/motiflet_' + amc_name + "_" + str(j) + '.gif'
            FuncAnimation(fig,
                          lambda i: draw_frame(ax, motions, joints, i,
                                               joints_to_highlight=filtered_joints),
                          range(pos, pos + motif_length, 4)).save(
                out_path,
                bitrate=100,
                fps=20)
