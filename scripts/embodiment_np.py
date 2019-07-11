#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotmat_z(angle):
    sina = np.sin(angle)
    cosa = np.cos(angle)
    return np.array([[cosa, -sina, 0, 0],
                     [sina, cosa, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def rotmat_z_dot(angle, angle_dot):
    sina = np.sin(angle)
    cosa = np.cos(angle)
    return np.array([[-sina * angle_dot, -cosa * angle_dot, 0, 0],
                     [cosa * angle_dot, -sina * angle_dot, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])


def plot_line_3d(gca, start, end, color='black', linewidth=1.0):
    xs = [start[0], end[0]]
    ys = [start[1], end[1]]
    zs = [start[2], end[2]]
    gca.plot(xs, ys, zs=zs, zdir='z', color=color, linewidth=linewidth)


def plot_frame_axes(gca, frame, scale_factor=0.075):
    frame = np.array(frame)[0:3, :]
    origin = frame[:, 3]
    plot_line_3d(gca, origin, origin + scale_factor * frame[:, 0], 'red', 2.0)
    plot_line_3d(gca, origin, origin + scale_factor * frame[:, 1], 'green')
    plot_line_3d(gca, origin, origin + scale_factor * frame[:, 2], 'blue')


class Embodiment:
    def __init__(self, relative_joint_frames, parent_ids, relative_link_frames=None, angle_ranges=None, velocity_ranges=None, effort_ranges=None):
        assert parent_ids[0] is None, "First element of learner_parent_ids must me 'None'!"
        for i in range(1, len(parent_ids)):
            assert parent_ids[i] < i, "Elements in learner_parent_ids can only refer to elements to the left!"

        self.relative_joint_frames = relative_joint_frames
        self.num_links = len(parent_ids)
        self.parent_ids = parent_ids

        self.chain_length, self.chain_lengths = self._longest_chain_length()
        self.normalization_factor = 1.0 / self.chain_length
        self.relative_joint_frames_normalized = self.relative_joint_frames.copy()
        self.relative_joint_frames_normalized[:, 0:3, 3] *= self.normalization_factor

        if relative_link_frames is None:
            self.relative_link_frames = np.array([np.eye(4, 4) for _ in range(len(parent_ids))])
        else:
            self.relative_link_frames = relative_link_frames
        self.relative_link_frames_normalized = self.relative_link_frames.copy()
        self.relative_link_frames_normalized[:, 0: 3, 3] *= self.normalization_factor

        if angle_ranges is None:
            self.angle_ranges = np.array([[-np.pi] * self.num_links,
                                          [np.pi] * self.num_links])
        else:
            assert len(angle_ranges[0]) is self.num_links
            assert len(angle_ranges[1]) is self.num_links
            self.angle_ranges = np.array(angle_ranges)
        self.angle_intervals = self.angle_ranges[1] - self.angle_ranges[0]

        if velocity_ranges is None:
            self.velocity_ranges = np.array([[-100] * self.num_links,
                                             [100] * self.num_links])
        else:
            assert len(velocity_ranges[0]) is self.num_links
            assert len(velocity_ranges[1]) is self.num_links
            self.velocity_ranges = np.array(velocity_ranges)

        if effort_ranges is None:
            self.effort_ranges = np.array([[-100] * self.num_links,
                                           [100] * self.num_links])
        else:
            assert len(effort_ranges[0]) is self.num_links
            assert len(effort_ranges[1]) is self.num_links
            self.effort_ranges = np.array(effort_ranges)

        self.link_dists_from_origin = self._link_positions_in_chain()

    def absolute_joint_frames(self, joint_angles, normalized=False):
        assert len(joint_angles) is self.num_links
        relative_transformations = [rotmat_z(angle) for angle in joint_angles]
        if normalized:
            relative_transformations = np.matmul(relative_transformations, self.relative_joint_frames_normalized)
        else:
            relative_transformations = np.matmul(relative_transformations, self.relative_joint_frames)

        absolute_transformations_joints = [relative_transformations[0]]
        for i in range(1, self.num_links):
            new_transformation = np.matmul(absolute_transformations_joints[self.parent_ids[i]], relative_transformations[i])
            absolute_transformations_joints.append(new_transformation)

        return np.array(absolute_transformations_joints), relative_transformations

    def absolute_joint_frames_batch(self, joint_angles_batch, normalized=False):
        return [self.absolute_joint_frames(joint_angles, normalized) for joint_angles in joint_angles_batch]

    def absolute_link_frames(self, absolute_joint_frames, normalized=False):
        if normalized:
            return np.matmul(absolute_joint_frames, self.relative_link_frames_normalized)
        else:
            return np.matmul(absolute_joint_frames, self.relative_link_frames)

    def absolute_link_frames_batch(self, absolute_joint_frames_batch, normalized=False):
        return [self.absolute_link_frames(absolute_joint_frames, normalized) for absolute_joint_frames in absolute_joint_frames_batch]

    def absolute_frames(self, joint_angles, normalized=False):
        absolute_joint_frames, _ = self.absolute_joint_frames(joint_angles, normalized)
        absolute_link_frames = self.absolute_link_frames(absolute_joint_frames, normalized)
        return absolute_joint_frames, absolute_link_frames

    def absolute_frames_batch(self, joint_angles_batch, normalized=False):
        absolute_joint_frames_batch = self.absolute_joint_frames_batch(joint_angles_batch, normalized)
        absolute_link_frames_batch = self.absolute_link_frames_batch(absolute_joint_frames_batch, normalized)
        return absolute_joint_frames_batch, absolute_link_frames_batch

    def data_matrices(self, joint_angles, joint_velocities, normalized=False):
        """
        Generates a list of matrices, each describing the current state of an embodiment's link.
        :param joint_angles: Joint angles in rad.
        :param joint_velocities: Joint velocities in rad/s.
        :param normalized: If 'True', the embodiment's normalized frames will be used.
        :return: A list of 3x4 matrices, each corresponding to the link frames, where each:
                    - first column contains the x-Axis of the link frame described in fixed-frame coordinates
                    - second column contains the frame-origin's linear velocity in fixed-frame coordinates
                    - third column contains the rotation axis and speed of the link
                    - fourth column contains the position of the link-frame's origin in fixed-frame coordinates
        """
        absolute_joint_frames, relative_joint_frames = self.absolute_joint_frames(joint_angles, normalized)
        absolute_link_frames = self.absolute_link_frames(absolute_joint_frames, normalized)

        relative_rdots = [rotmat_z_dot(angle, angle_dot) for angle, angle_dot in zip(joint_angles, joint_velocities)]
        if normalized:
            relative_tdots = np.matmul(relative_rdots, self.relative_joint_frames_normalized)
        else:
            relative_tdots = np.matmul(relative_rdots, self.relative_joint_frames)

        absolute_tdots = [relative_tdots[0]]
        for i in range(1, self.num_links):
            new_joint_transformation = np.matmul(absolute_tdots[self.parent_ids[i]], relative_joint_frames[i]) + \
                                       np.matmul(absolute_joint_frames[self.parent_ids[i]], relative_tdots[i])
            absolute_tdots.append(new_joint_transformation)

        if normalized:
            absolute_link_tdots = np.matmul(absolute_tdots, self.relative_link_frames_normalized)
        else:
            absolute_link_tdots = np.matmul(absolute_tdots, self.relative_link_frames)

        # TODO: Speedup by manual inversion?
        absolute_link_frames_inv = np.linalg.inv(absolute_link_frames)
        spacial_twists = np.matmul(absolute_link_tdots, absolute_link_frames_inv)

        data_matrices = []
        for i in range(self.num_links):
            angular_velocity = np.concatenate([[spacial_twists[i, 2, 1]],
                                               [spacial_twists[i, 0, 2]],
                                               [spacial_twists[i, 1, 0]]], 0)
            data_matrix = np.concatenate([np.transpose([absolute_link_frames[i, 0:3, 0]]),
                                          np.transpose([absolute_link_tdots[i, 0:3, 3]]),
                                          np.transpose([angular_velocity]),
                                          np.transpose([absolute_link_frames[i, 0:3, 3]])], 1)
            data_matrices.append(data_matrix)

        return data_matrices, absolute_joint_frames

    # TODO: Draw frames/link velocities?
    def plot(self, joint_angles, axes, normalized=False, link_frames=False):
        """
        Drawing function that plots the embodiment in given axes. Assumes a non-branched chain!
        :param joint_angles: Joint angles of configuration to plot.
        :param axes: The PyPlot axes object to plot the embodiment in.
        :param normalized: Flag if normalized coordinates should be used.
        :return: None
        """
        absolute_joint_frames, absolute_link_frames = self.absolute_frames(joint_angles, normalized)
        xs = np.concatenate([[0], absolute_joint_frames[:, 0, 3]])
        ys = np.concatenate([[0], absolute_joint_frames[:, 1, 3]])
        zs = np.concatenate([[0], absolute_joint_frames[:, 2, 3]])
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")

        axes.plot(xs, ys, 'o-', zs=zs, zdir='z', color='black')

        for joint_frame in absolute_joint_frames:
            plot_frame_axes(axes, joint_frame, 0.03)

        if link_frames:
            for link_frame in absolute_link_frames:
                # start = link_frame[0:3, 3]
                # end = link_frame[0:3, 3] + 0.01 * link_frame[0:3, 0]
                # axes.plot([start[0], end[0]], [start[1], end[1]], 'o-', zs=[start[2], end[2]], zdir='z', color='blue')
                plot_frame_axes(axes, link_frame)

    def show(self, joint_angles, normalized=False, link_frames=False):
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        self.plot(joint_angles, axes, normalized, link_frames)
        axes.set_xlim3d(-0.95, 0.95)
        axes.set_ylim3d(-0.95, 0.95)
        axes.set_zlim3d(-0.1, 1)
        plt.show()

    def _longest_chain_length(self):
        lengths = [np.linalg.norm(self.relative_joint_frames[i, 0:3, 3]) for i in range(self.num_links)]
        chain_lengths = [lengths[0]]
        for i in range(1, self.num_links):
            chain_lengths.append(chain_lengths[self.parent_ids[i]] + lengths[i])
        return np.max(chain_lengths), chain_lengths

    def _link_positions_in_chain(self):
        normalized_chain_lengths = np.array(self.chain_lengths) * self.normalization_factor
        dists_from_joints = [np.linalg.norm(self.relative_link_frames_normalized[i, 0:3, 3]) for i in
                             range(self.num_links)]
        dists_from_origin = normalized_chain_lengths - np.array(dists_from_joints)
        return dists_from_origin

    @staticmethod
    def create_n_link_embodiment(n, link_offset=0.5):
        link_length = 1.0 / n
        relative_joint_frames = np.array([[[1, 0, 0, link_length],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]]] * n)
        relative_link_frames = np.array([[[1, 0, 0, -link_length * link_offset],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]]] * n)
        parent_ids = [None] + list(range(n - 1))
        # print(relative_joint_frames)
        # print(relative_link_frames)

        return Embodiment(relative_joint_frames, parent_ids, relative_link_frames)
