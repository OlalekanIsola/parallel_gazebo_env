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
panda_angle_ranges = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                      [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]]
panda_velocity_ranges = [[-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100],
                         [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]]
panda_effort_ranges = [[-87, -87, -87, -87, -12, -12, -0.01],
                       [87, 87, 87, 87, 12, 12, 0.01]]

panda_4j_parent_ids = [None, 0, 1, 2]
panda_4j_relative_joint_frames = np.array([panda_relative_joint_frames[0],
                                           np.matmul(panda_relative_joint_frames[1], panda_relative_joint_frames[2]),
                                           panda_relative_joint_frames[3],
                                           np.matmul(np.matmul(panda_relative_joint_frames[4], panda_relative_joint_frames[5]), panda_relative_joint_frames[6])],
                                          dtype='float32')
# Frames that define the link, each as seen from the joint frame to the next link
# x-axis should point along main line of link towards the next link
panda_4j_relative_link_frames = np.array([[[0, 1, 0, 0],
                                           [-1, 0, 0, 0.12],
                                           [0, 0, 1, -0.01],
                                           [0, 0, 0, 1]],

                                          [[np.cos(80 * np.pi / 180), -np.sin(80 * np.pi / 180), 0, -0.0525],
                                           [np.sin(80 * np.pi / 180), np.cos(80 * np.pi / 180), 0, -0.15],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          panda_relative_link_frames[3],

                                          [[np.cos(80 * np.pi / 180), 0, np.sin(80 * np.pi / 180), -0.07],
                                           [0, 1, 0, 0],
                                           [-np.sin(80 * np.pi / 180), 0, np.cos(80 * np.pi / 180), 0.0],
                                           [0, 0, 0, 1]]], dtype='float32')
panda_4j_angle_ranges = [[-2.8973, -1.7628, -3.0718, -2.8973],
                         [2.8973, 1.7628, -0.0698, 2.8973]]
panda_4j_velocity_ranges = [[-2.1750, -2.1750, -2.1750, -2.6100],
                            [2.1750, 2.1750, 2.1750, 2.6100]]
panda_4j_effort_ranges = [[-87, -87, -87, -12],
                       [87, 87, 87, 12]]

panda_3j_parent_ids = [None, 0, 1]
panda_3j_relative_joint_frames = np.array([np.matmul(panda_relative_joint_frames[0], np.matmul(panda_relative_joint_frames[1], panda_relative_joint_frames[2])),
                                           panda_relative_joint_frames[3],
                                           np.matmul(np.matmul(panda_relative_joint_frames[4], panda_relative_joint_frames[5]), panda_relative_joint_frames[6])],
                                          dtype='float32')
# Frames that define the link, each as seen from the joint frame to the next link
# x-axis should point along main line of link towards the next link
panda_3j_relative_link_frames = np.array([[[np.cos(80 * np.pi / 180), -np.sin(80 * np.pi / 180), 0, -0.0525],
                                           [np.sin(80 * np.pi / 180), np.cos(80 * np.pi / 180), 0, -0.15],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],

                                          panda_relative_link_frames[3],

                                          [[np.cos(80 * np.pi / 180), 0, np.sin(80 * np.pi / 180), -0.07],
                                           [0, 1, 0, 0],
                                           [-np.sin(80 * np.pi / 180), 0, np.cos(80 * np.pi / 180), 0.0],
                                           [0, 0, 0, 1]]], dtype='float32')
panda_3j_angle_ranges = [[-2.8973, -3.0718, -2.8973],
                         [2.8973, -0.0698, 2.8973]]
panda_3j_velocity_ranges = [[-2.1750, -2.1750, -2.6100],
                            [2.1750, 2.1750, 2.6100]]
panda_3j_effort_ranges = [[-87, -87, -12],
                          [87, 87, 12]]

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

panda_embodiment = embodiment.Embodiment(panda_relative_joint_frames, panda_parent_ids, panda_relative_link_frames, panda_angle_ranges, panda_velocity_ranges, panda_effort_ranges)
panda_embodiment.link_dists_from_origin = [0.16057740356113898, 0.3414777004183537, 0.4639223628787724, 0.5403012448100774, 0.7034305162879936, 0.8201778297256058, 0.8854359750540293]
panda_4j_embodiment = embodiment.Embodiment(panda_4j_relative_joint_frames, panda_4j_parent_ids, panda_4j_relative_link_frames, panda_4j_angle_ranges, panda_4j_velocity_ranges, panda_4j_effort_ranges)
panda_4j_embodiment.link_dists_from_origin = [0.16057740356113898, 0.4, 0.5403012448100774, 0.72]
panda_3j_embodiment = embodiment.Embodiment(panda_3j_relative_joint_frames, panda_3j_parent_ids, panda_3j_relative_link_frames, panda_3j_angle_ranges, panda_3j_velocity_ranges, panda_3j_effort_ranges)
panda_3j_embodiment.link_dists_from_origin = [0.27, 0.5403012448100774, 0.72]
branched_embodiment = embodiment.Embodiment(branched_relative_joint_frames, branched_parent_ids, branched_relative_link_frames)
twolink_embodiment = embodiment.Embodiment(twolink_relative_joint_frames, twolink_parent_ids, twolink_relative_link_frames)
twolink2_embodiment = embodiment.Embodiment(twolink2_relative_joint_frames, twolink_parent_ids, twolink2_relative_link_frames)
threelink_embodiment = embodiment.Embodiment(threelink_relative_joint_frames, threelink_parent_ids, threelink_relative_link_frames)
threelink2_embodiment = embodiment.Embodiment(threelink2_relative_joint_frames, threelink_parent_ids, threelink2_relative_link_frames)
fourlink_embodiment = embodiment.Embodiment(fourlink_relative_joint_frames, fourlink_parent_ids, fourlink_relative_link_frames)


if __name__ == '__main__':
    angles = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    # panda_embodiment.show(angles, link_frames=True)
    angles_4j = [-1.0, -1.0, -1.0, -1.0]
    angles_4j0 = [[0.0, 0.0, 2.0, 0.0],
                  [0.0, 0.0, 2.0, 1.0],
                  [0.0, 0.0, 2.0, 2.0],
                  [0.0, 0.0, 2.0, 3.0]]
    angles_3j = [-1.0, -1.0, -1.0]
    panda_3j_embodiment.show(angles_3j)
    # for angles4j in angles_4j0: panda_4j_embodiment.show(angles4j, link_frames=True)

    # joint_frames_panda, _ = panda_embodiment.absolute_joint_frames(angles, False)
    # joint_frames_panda_4j, _ = panda_4j_embodiment.absolute_joint_frames(angles_4j, False)
    # print(joint_frames_panda)
    # print(joint_frames_panda_4j)


    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d', proj_type='ortho')
    # panda_embodiment.plot(angles, ax1)
    # plt.show()