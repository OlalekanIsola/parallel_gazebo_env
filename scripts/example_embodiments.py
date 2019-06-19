#! /usr/bin/env python3

import numpy as np
import embodiment_np as embodiment

panda_parent_ids = [None, 0, 1, 2, 3, 4, 5]
panda_relative_joint_frames = np.array([[[1, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, -1, 0, 0.33],
                                         [0, 0, 0, 1]],

                                        [[1, 0, 0, 0],
                                         [0, 0, -1, -0.316],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]],

                                        [[1, 0, 0, 0.0825],
                                         [0, 0, -1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]],

                                        [[1, 0, 0, -0.0825],
                                         [0, 0, 1, 0.384],
                                         [0, -1, 0, 0],
                                         [0, 0, 0, 1]],

                                        [[1, 0, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]],

                                        [[1, 0, 0, 0.088],
                                         [0, 0, -1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]],

                                        [[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0.1],
                                         [0, 0, 0, 1]]], dtype='float32')
# Frames that define the link, each as seen from the joint frame to the next link
# x-axis should point along main line of link towards the next link
panda_relative_link_frames = np.array([[[0, 1, 0, 0],
                                        [-1, 0, 0, 0.12],
                                        [0, 0, 1, -0.01],
                                        [0, 0, 0, 1]],

                                       [[0, 0, -1, 0],
                                        [0, 1, 0, 0.01],
                                        [1, 0, 0, -0.2],
                                        [0, 0, 0, 1]],

                                       [[np.cos(70*np.pi/180), -np.sin(70*np.pi/180), 0, -0.0825],
                                        [np.sin(70*np.pi/180), np.cos(70*np.pi/180), 0, -0.04],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]],

                                       [[np.cos(-110*np.pi/180), 0, np.sin(-110*np.pi/180), 0.06],
                                        [0, 1, 0, 0],
                                        [-np.sin(-110*np.pi/180), 0, np.cos(-110*np.pi/180), -0.344],
                                        [0, 0, 0, 1]],

                                       [[0, -1, 0, 0],
                                        [1, 0, 0, -0.14],
                                        [0, 0, 1, -0.02],
                                        [0, 0, 0, 1]],

                                       [[0.866, 0, -0.5, -0.03],
                                        [0, 1, 0, 0],
                                        [0.5, 0, 0.866, 0],
                                        [0, 0, 0, 1]],

                                       [[0, 0, -1, 0],
                                        [0, 1, 0, 0],
                                        [1, 0, 0, -0.02],
                                        [0, 0, 0, 1]]], dtype='float32')

branched_parent_ids = [None, 0, 0, 2, 3, 4, 3]
branched_relative_joint_frames = np.array([[[1, 0, 0, 0.4],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],

                                           [[1, 0, 0, 0.4],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],

                                           [[1, 0, 0, 0.4],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],

                                           [[1, 0, 0, 0.4],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],

                                           [[1, 0, 0, 0.4],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],

                                           [[1, 0, 0, 0.2],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],

                                           [[1, 0, 0, 0.8],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]]], dtype='float32')

branched_relative_link_frames = np.array([[[1, 0, 0, -0.2],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          [[1, 0, 0, -0.2],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          [[1, 0, 0, -0.2],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          [[1, 0, 0, -0.2],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          [[1, 0, 0, -0.2],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          [[1, 0, 0, -0.1],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          [[1, 0, 0, -0.4],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]]], dtype='float32')

twolink_parent_ids = [None, 0]
twolink_relative_joint_frames = np.array(
    [[[1., 0., 0., 5.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 5.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')
twolink_relative_link_frames = np.array(
    [[[1., 0., 0., -2.5], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -2.5], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')

twolink2_relative_joint_frames = np.array(
    [[[1., 0., 0., 4.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 3.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')
twolink2_relative_link_frames = np.array(
    [[[1., 0., 0., -2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.5], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')

threelink_parent_ids = [None, 0, 1]
threelink_relative_joint_frames = np.array(
    [[[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')
threelink_relative_link_frames = np.array(
    [[[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')
threelink2_relative_joint_frames = np.array(
    [[[1., 0., 0., 1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')
threelink2_relative_link_frames = np.array(
    [[[1., 0., 0., -0.5], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -0.5], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')

fourlink_parent_ids = [None, 0, 1, 2]
fourlink_relative_joint_frames = np.array(
    [[[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., 2.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')
fourlink_relative_link_frames = np.array(
    [[[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
     [[1., 0., 0., -1.0], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
    dtype='float32')

panda_embodiment = embodiment.Embodiment(panda_relative_joint_frames, panda_parent_ids, panda_relative_link_frames)
panda_embodiment.link_dists_from_origin = [0.16057740356113898, 0.3414777004183537, 0.4639223628787724, 0.5403012448100774, 0.7034305162879936, 0.8201778297256058, 0.8854359750540293]
branched_embodiment = embodiment.Embodiment(branched_relative_joint_frames, branched_parent_ids, branched_relative_link_frames)
twolink_embodiment = embodiment.Embodiment(twolink_relative_joint_frames, twolink_parent_ids, twolink_relative_link_frames)
twolink2_embodiment = embodiment.Embodiment(twolink2_relative_joint_frames, twolink_parent_ids, twolink2_relative_link_frames)
threelink_embodiment = embodiment.Embodiment(threelink_relative_joint_frames, threelink_parent_ids, threelink_relative_link_frames)
threelink2_embodiment = embodiment.Embodiment(threelink2_relative_joint_frames, threelink_parent_ids, threelink2_relative_link_frames)
fourlink_embodiment = embodiment.Embodiment(fourlink_relative_joint_frames, fourlink_parent_ids, fourlink_relative_link_frames)


if __name__ == '__main__':
    angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    panda_embodiment.show(angles)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d', proj_type='ortho')
    panda_embodiment.plot(angles, ax1)
    plt.show()