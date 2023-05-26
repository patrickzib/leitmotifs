import matplotlib
import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import amc.amc_parser as amc_parser

from matplotlib.animation import FuncAnimation

from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as sch


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


def draw_frame(ax, motions, joints, i, joints_to_highlight=None):
    ax.cla()
    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    joints['root'].set_motion(motions[i])

    c_joints = joints['root'].to_dict()
    xs, ys, zs = [], [], []
    for joint in c_joints.values():
        xs.append(joint.coordinate[0, 0])
        ys.append(joint.coordinate[1, 0])
        zs.append(joint.coordinate[2, 0])

    ax.plot(zs, xs, ys, 'b.')

    for joint in c_joints.values():
        child = joint
        if child.parent is not None:
            parent = child.parent
            xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
            ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
            zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
            ax.plot(zs, xs, ys, 'r')



# plot multi-variat motiflet
def plot_multivariate_motiflet(
        data, motifset, m, d=[], names=[]
    ):

    fig, axes = plt.subplots(len(data) + 1, 1, figsize=(14, 2*len(data)))

    for i in range(len(data)):
        ax = axes[i]
        # plt.subplot(len(data) + 1, 1, i + 1)
        if len(names) == len(data):
            ax.set_title('${{{0}}}$'.format(names[i]))
        else:
            ax.set_title('$T_{{{0}}}$'.format(i + 1))

        for idx, pos in enumerate(motifset):
            ax.plot(range(0, m), data[i, :][pos:pos + m]) #c=color[idx])

        ax.set_xlim((0, m))

    plt.tight_layout()
    plt.show()


# http://mocap.cs.cmu.edu/search.php?subjectnumber=13&motion=%

# ks = 15
# motif_length = 100
# amc_name = "13_17" # Boxing
# asf_path = '../datasets/motion_data/13.asf'


ks = 15
motif_length = 120
amc_name = "93_08" # Fancy Charleston
# amc_name = "93_04" # Side By Side Female
# amc_name = "93_05" # Side By Side Male
asf_path = '../datasets/motion_data/93.asf'



amc_path = '../datasets/motion_data/'+amc_name+'.amc'

#use_joints = np.asarray(
#    ['root', 'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head',
#     'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
#     'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
#     'rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes'])

# Body
# use_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']

# Right
#use_joints = ['rclavicle', 'rhumerus', 'rradius', 'rwrist',
              #'rhand', 'rfingers', 'rthumb']

#use_joints = [  'lhand', 'lfingers', 'lthumb'
#               'rhand', 'rfingers', 'rthumb']

use_joints = [  'rclavicle', 'rhumerus', 'rradius', 'rwrist',
                'rhand', 'rfingers', 'rthumb',
                'rfemur', 'rtibia', 'rfoot', 'rtoes']

# footwork
# use_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']


def test_plot_length_selection():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, use_joints)

    print("Used joints:", use_joints)
    print("Data", df.shape)

    series = df.values

    ks = 15
    length_range = list(range(10, 200, 10))
    print(length_range)

    m = plot_motif_length_selection(
        ks,
        series,
        ds_name=amc_name,
        motif_length_range=length_range,
        slack=1.0)

    print("----")
    print("Best length", m)
    print("----")


def test_motion_capture():
    generate_motion_capture(use_joints)


def generate_motion_capture(joints_to_use, prefix=None, add_xyz=True):
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, joints_to_use, add_xyz=add_xyz)

    print("Used joints:", joints_to_use)
    series = df.values

    ks = 15
    motif_length = 120

    #dists, candidates, elbow_points, m = ml.search_k_motiflets_elbow(
    #    ks,
    #    series,
    #    slack=0.5,
    #    motif_length=motif_length)

    dists, candidates, elbow_points = plot_elbow(
        ks, series,
        ds_name=amc_name,
        slack=0.5,
        plot_elbows=True,
        motif_length=motif_length)

    print("----")
    print(dists)
    print(elbow_points)
    print(list(candidates[elbow_points]))
    print("----")

    motiflets = candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        for j, pos in enumerate(motiflet):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            if prefix:
                out_path = 'video/motiflet_'+amc_name+'_'+prefix+'_'\
                           +str(i)+'_'+str(j)+'.gif'
            else:
                out_path = 'video/motiflet_' + amc_name + '_' \
                           + str(i) + '_' + str(j) + '.gif'

            FuncAnimation(fig,
                          lambda i: draw_frame(ax, motions, joints, i),
                          range(pos, pos+motif_length, 4)).save(
                                out_path,
                                bitrate=1000,
                                fps=20)


def tests():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, use_joints)

    print("Used joints:", use_joints)
    series = df.values
    D_ = ml.compute_distances_full_mv(series, m=100, slack=1.0)

    dim = 12
    best = np.argpartition(D_, dim, axis=0)[:dim]
    D_ = np.take_along_axis(D_, best, axis=0)
    print(best.shape)

    print("done")


def test_plotting():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, use_joints)

    print("Used joints:", use_joints)
    series = df.values

    ks = 15
    motif_length = 120

    dists, motiflets, elbow_points = plot_elbow(
        ks, series,
        ds_name=amc_name,
        slack=0.5,
        plot_elbows=True,
        motif_length=motif_length)

    print("----")
    print(dists)
    print(elbow_points)
    print(list(motiflets[elbow_points]))
    print("----")


def test_dimension_plotting():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, use_joints)

    joints = df.index
    print("Used joints:", use_joints)
    series = df.values

    ks = 15
    motif_length = 120

    dists, motiflets, elbow_points = plot_elbow_by_dimension(
        ks, series,
        dimension_labels=joints,
        ds_name=amc_name,
        slack=0.5,
        motif_length=motif_length)

    print("----")

    series = np.zeros((df.shape[0], df.shape[1] - motif_length), dtype=np.float32)
    for i in range(series.shape[0]):
        for pos in motiflets[i, elbow_points[i][-1]]:
            series[i, pos:pos+motif_length] = 1

    X = series  # zscore(series, axis=1)

    # size of image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    Z = sch.linkage(X, method='ward')

    # creating the dendrogram
    dend = sch.dendrogram(
        Z, labels=joints, ax = ax)

    plt.axhline(y=127.5, color='orange')
    ax.set_title('Dendrogram')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Euclidean distances')
    plt.tight_layout()
    plt.show()

    k = 3
    y_dimensions = sch.fcluster(Z, k, criterion='maxclust')
    mapping = list(zip(y_dimensions, joints))

    joint_clusters = {}
    for i in range(1,k+1):
        print("Cluster", i)
        joint_clusters[i] = [x[1] for x in mapping if x[0] == i]
        print(joint_clusters[i])
        print("----")

        generate_motion_capture(joint_clusters[i],
                                prefix="Cluster"+str(i), add_xyz=False)


    # joint_clusters = {1: }


def test_filter():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    # df = exclude_body_joints(df)
    # df = include_joints(df, use_joints)

    to_use = ['rfemur_y', 'rtibia_x', 'rtibia_y', 'rfoot_x', 'rtoes_x']
    print(include_joints(df, to_use, add_xyz=False))






