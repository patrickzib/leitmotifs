import matplotlib
import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import amc.amc_parser as amc_parser

from matplotlib.animation import FuncAnimation



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
    for k, v in zip('xyz', exclude):
        exclude_bones.extend([x + "_" + k for x in exclude])
    exclude_bones

    return df[~df.index.isin(exclude_bones)]

def include_joints(df, include):
    include_bones = []
    for k, v in zip('xyz', include):
        include_bones.extend([x + "_" + k for x in include])

    return df[df.index.isin(include_bones)]


def draw_frame(ax, motions, joints, i):
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


amc_name = "13_17"
asf_path = '../datasets/motion_data/13.asf'
amc_path = '../datasets/motion_data/'+amc_name+'.amc'

#amc_name = "19_15"
#asf_path = '../datasets/motion_data/19.asf'
#amc_path = '../datasets/motion_data/'+amc_name+'.amc'


# http://mocap.cs.cmu.edu/search.php?subjectnumber=13&motion=%
# 13_29: jumping jacks, side twists, bend over, squats

#joint_names = np.asarray(
#    ['root', 'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head',
#     'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
#     'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
#     'rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes'])

# Body
# use_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']

# Right
#use_joints = [  'rclavicle', 'rhumerus', 'rradius', 'rwrist',
#               'rhand', 'rfingers', 'rthumb']

use_joints = [  'lhand', 'lfingers', 'lthumb'
               'rhand', 'rfingers', 'rthumb']

#use_joints = [  'rclavicle', 'rhumerus', 'rradius', 'rwrist',
#                'rhand', 'rfingers', 'rthumb',
#                'rfemur', 'rtibia', 'rfoot', 'rtoes']

# footwork
# use_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']

def test_motion_capture():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, use_joints)

    # bones = df.index
    series = df.values

    ks = 10
    motif_length = 100

    dists, candidates, elbow_points, m = ml.search_k_motiflets_elbow(
        ks,
        series,
        slack=1.0,
        motif_length=motif_length)

    print("----")
    print(list(candidates[elbow_points]))
    print("----")

    motiflets = candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        for j, pos in enumerate(motiflet):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            out_path = 'video/motiflet_'+amc_name+'_'+str(i)+'_'+str(j)+'.gif'
            FuncAnimation(fig,
                          lambda i: draw_frame(ax, motions, joints, i),
                          range(pos, pos+motif_length, 4)).save(
                                out_path,
                                bitrate=1000,
                                fps=20)


# def test_animate():
#
#     motif_length = 100
#     motif_sets = [np.array([4463, 1747, 1067,  233], dtype=np.int32),
#                   np.array([ 792, 1893, 2903, 3575, 1848, 2253, 2499, 3805], dtype=np.int32)]
#
#     joints = amc_parser.parse_asf(asf_path)
#     motions = amc_parser.parse_amc(amc_path)
#
#     for i, motiflet in enumerate(motif_sets):
#         for j, pos in enumerate(motiflet):
#             fig = plt.figure()
#             ax = plt.axes(projection='3d')
#             # ax = Axes3D(fig)
#
#             out_path = 'video/motiflet_'+str(i)+'_'+str(j)+'.gif'
#             FuncAnimation(fig,
#                           lambda i: draw_frame(ax, motions, joints, i),
#                           range(pos, pos+motif_length, 4)).save(
#                                 out_path,
#                                 bitrate=1000,
#                                 fps=20)
#
#     plt.close('all')
#
# def test_plot_motiflet_2():
#     motif_length = 50
#     motif_sets = [np.array([4463, 1747, 1067,  233], dtype=np.int32),
#                   np.array([ 792, 1893, 2903, 3575, 1848, 2253, 2499, 3805], dtype=np.int32)]
#
#     for motiflet_pos in motif_sets:
#
#         joints = amc_parser.parse_asf(asf_path)
#         motions = amc_parser.parse_amc(amc_path)
#
#         df = pd.DataFrame(
#             [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
#         df = exclude_body_joints(df)
#         df = include_joints(df, use_joints)
#
#         bones = df.index
#         series = df.values
#
#         plot_multivariate_motiflet(series, motiflet_pos, motif_length, names=bones)

def tests():

    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, use_joints)

    data = df.values
    D_ = ml.compute_distances_full_mv(data, m=50, slack=1.0)

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

    ks = 10
    motif_length = 100

    dists, motiflets, elbow_points = plot_elbow(
        ks, series,
        ds_name=amc_name,
        plot_elbows=True,
        motif_length=motif_length)


    print("----")
    print(list(motiflets[elbow_points]))
    print("----")
