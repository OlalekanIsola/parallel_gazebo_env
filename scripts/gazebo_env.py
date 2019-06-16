#!/usr/bin/env python3

# import sys
# import copy
import rospy
import rosbag
import gym
import csv
import time
import numpy as np
from scipy import special
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
import example_embodiments

# TODO: NaN prevention -> include recent action in state? RNN?
# TODO: Use timestamp from last joint_state to get bag message?
# TODO: Short Motion after resetting?
# TODO: Parallel environments


DEBUG = True


def link_distance(data_matrix1, data_matrix2):
    """
    Uses the information from the given data matrices to calculate the distance in linear/angular position/motion of the
    frames, described by the data matrices. For reference of the format see embodiment_np.Embodiment.data_matrices().
    :param data_matrix1: Data matrix from link 1
    :param data_matrix2: Data matrix from link 2
    :return: The distance measure between the two frames (scalar).
    """
    weight_o_dist = 1.0
    weight_lin_vel = 0.1
    weight_rot_vel = 0.1

    translation_distance = np.linalg.norm(data_matrix1[:, 3] - data_matrix2[:, 3])

    # For orientation distance, the angle between the x-axes of the link frames are used, which by convention should
    # coincide with the direction of the robot link they describe. As approximation of the angle, the cos of the angle,
    # that is the scalar product of the vectors (len = 1) is being used.
    orientation_distance = 1 - np.dot(data_matrix1[:, 0], data_matrix2[:, 0])

    # TODO: Distances of velocities other than euclidean?
    lin_vel_distance = np.linalg.norm(data_matrix1[:, 1] - data_matrix2[:, 1])
    rot_vel_distance = np.linalg.norm(data_matrix1[:, 2] - data_matrix2[:, 2])

    #return translation_distance + weight_o_dist * orientation_distance + weight_lin_vel * lin_vel_distance + weight_rot_vel * rot_vel_distance
    return weight_o_dist * orientation_distance + weight_lin_vel * lin_vel_distance + weight_rot_vel * rot_vel_distance


def cartesian_product(list1, list2, flat=True):
    """
    Calculates the cartesian product of two lists.
    :param list1: The first list, size m.
    :param list2: The second list, size n.
    :param flat: If True, all combinations will be arranged along the first dimension, if False, they will be arranged
                 in a m x n grid.
    :return: A combination of each element of list1 with each element of list2, either in a flat or grid form.
    """
    assert np.shape(list1)[1:] == np.shape(list2)[1:], "The elements of each list do not have the same dimensions!"

    list1_repeated = np.expand_dims(np.repeat(list1, len(list2), 0), 1)
    reps = np.ones_like(np.shape(list2))
    reps[0] = len(list1)
    list2_tiled = np.expand_dims(np.tile(list2, reps), 1)

    flat_cartesian_product = np.concatenate([list1_repeated, list2_tiled], 1)

    if flat:
        return flat_cartesian_product
    else:
        desired_shape = [len(list1), len(list2), 2]
        desired_shape.extend(np.shape(list1)[1:])
        return np.reshape(flat_cartesian_product, desired_shape)


def calculate_weight_matrix(e_embodiment, l_embodiment):
    """
    Compute the weight matrix that asymetrically assigns the links of each embodiment to links of the other embodiment.
    :param e_embodiment: The expert embodiment.
    :param l_embodiment: The learner embodiment.
    :return: A e_n_links x l_n_links matrix with weights.
    """
    cart_product_chain_positions = np.transpose([np.repeat(e_embodiment.link_dists_from_origin,
                                                           l_embodiment.num_links),
                                                 np.tile(l_embodiment.link_dists_from_origin,
                                                         e_embodiment.num_links)])
    distances = np.abs(cart_product_chain_positions[:, 0] - cart_product_chain_positions[:, 1])
    distance_matrix = np.reshape(distances, [e_embodiment.num_links, l_embodiment.num_links])
    # TODO: Use real maximum instead of softmax?
    # argmaxes_el = np.argmin(distance_matrix, 0)
    # argmaxes_le = np.argmin(distance_matrix, 1)
    # weight_matrix = np.zeros(e_embodiment.num_links, l_embodiment.num_links)
    sm_el = special.softmax(-distance_matrix, 0)
    sm_le = special.softmax(-distance_matrix, 1)
    weight_matrix = sm_el + sm_le
    return weight_matrix


class GazeboEnvFullPanda(gym.Env):
    """
    An openAI gym environment to learn how to copy a robot motion.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size, duration, bagfile, e_embodiment, l_embodiment):
        super(GazeboEnvFullPanda, self).__init__()
        self.current_step = 0
        self.done = False
        self.step_size = step_size
        self.duration = duration
        self._received_first_ldata = False

        self.e_embodiment = e_embodiment
        self.l_embodiment = l_embodiment
        self.weight_matrix = calculate_weight_matrix(e_embodiment, l_embodiment)
        self.e_position = [0.0] * e_embodiment.num_links
        self.e_velocity = [0.0] * e_embodiment.num_links
        self.l_position = [0.0] * l_embodiment.num_links
        self.l_velocity = [0.0] * l_embodiment.num_links

        self.bagfile = rosbag.Bag(bagfile)
        self.bagfile_start_time = rospy.Time(self.bagfile.get_start_time())
        if DEBUG: print("Bagfile opened.")

        rospy.init_node('gym_environment_wrapper')
        if DEBUG: print("ROS node initialized.")

        self._joint_states_subscriber = rospy.Subscriber('panda1/joint_states', JointState, self._joint_state_callback, queue_size=1)
        if DEBUG: print("Joint State Subscriber registered.")

        self._command_publisher = rospy.Publisher('panda1/effort_jointgroup_controller/command', Float64MultiArray, queue_size=1)
        if DEBUG: print("Command Publisher registered.")
        self._command = Float64MultiArray()
        self._command_zero = Float64MultiArray()
        self._command.data = [0.0] * l_embodiment.num_links
        self._command_zero.data = [0.0] * l_embodiment.num_links

        self._pause_gazebo_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_gazebo_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset_gazebo_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._switch_controller_service = rospy.ServiceProxy('panda1/controller_manager/switch_controller', SwitchController)
        if DEBUG: print("Services registered.")

        # TODO: Find good value for max reward
        self.reward_range = (-5, 0)

        # Joint torque ranges
        self.action_space = gym.spaces.Box(
            low=np.array([-87, -87, -87, -87, -12, -12, -12]),
            high=np.array([87, 87, 87, 87, 12, 12, 12]),
            dtype='float32')

        # Respectively: l_joint_angles, l_joint_vels, t_joint_angles, t_joint_vels
        self.observation_space = gym.spaces.Box(
            low=np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                          [-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100],
                          [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                          [-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]]),
            high=np.array([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                           [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
                           [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                           [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]]),
            dtype='float32')

    def __del__(self):
        if DEBUG: print("Destructor.")
        self.bagfile.close()

    def reset(self):
        """
        Resets the environment. In order for the ROS controller to function properly after the reset, the desired effort
        will to be set to zero for all joints beforehand. After resetting it waits to receive the first joint_state
        message to reset the state of the learner.
        :return: Current observation.
        """
        if DEBUG: print("Reset method start.")
        self._command_publisher.publish(self._command_zero)
        if DEBUG: print("Zero command published.")
        self._reset_gazebo_service()
        if DEBUG: print("Gazebo reset.")
        self._unpause_gazebo_service()
        if DEBUG: print("Gazebo unpaused.")
        self._restart_joint_state_controller()
        if DEBUG: print("Joint State Controller resetted.")
        self._received_first_ldata = False
        while self._received_first_ldata is False:
            # TODO: better a sleep function? Resolve issues with ROS clock?
            pass
        self._pause_gazebo_service()
        if DEBUG: print("Gazebo paused.")
        self.current_step = 0
        try:
            self.e_position, self.e_velocity = self._get_expert_state_from_bagfile(self.current_step)
            self.done = False
        except StopIteration:
            if DEBUG: print("StopIteration Exception! Setting done->False")
            self.done = True
        if DEBUG: print("Reset method end.")
        self._command.data = [0.0] * self.l_embodiment.num_links
        return [self.l_position, self.l_velocity, self.e_position, self.e_velocity]

    def step(self, action):
        """
        Executes the given action by publishing the joint effort command to the ROS controller. In order to receive the
        joint_states messages properly, the joint_state_controller needs to be restarted.
        :param action: A list of joint efforts to send to the joints of the learner.
        :return: The current state/observation, the immediate reward and the 'done' flag.
        """
        if DEBUG: print("Running step with: {}".format(action))
        command_difference = [np.abs(action_val - last_action_val) for action_val, last_action_val in zip(action, self._command.data)]
        max_command_difference = np.max(command_difference)
        if max_command_difference >= 0.9:
            if DEBUG: print("Step: {} Received too large difference in effort command! Ending episode, robot destroyed.".format(self.current_step))
            reward  = -5
            observation = [self.l_position, self.l_velocity, self.e_position, self.e_velocity]
            return observation, reward, True, {}

        self._unpause_gazebo_service()
        if DEBUG: print("Unpaused Gazebo.")
        self._restart_joint_state_controller()
        if DEBUG: print("Restarted joint state controller.")
        self._command.data = action
        self._command_publisher.publish(self._command)
        if DEBUG: print("Published action.")
        rospy.sleep(self.step_size)
        if DEBUG: print("Slept.")
        self._pause_gazebo_service()
        if DEBUG: print("Paused Gazebo.")
        self.current_step += 1
        done = False

        try:
            self.e_position, self.e_velocity = self._get_expert_state_from_bagfile(self.current_step)
        except StopIteration:
            if DEBUG: print("StopIteration Exception! Setting done->False")
            done = True

        reward = self._calculate_reward(self.e_position, self.e_velocity, self.l_position, self.e_velocity)
        if DEBUG: print("Reward: {}".format(reward))
        observation = [self.l_position, self.l_velocity, self.e_position, self.e_velocity]

        return observation, reward, done, {}

    def render(self, mode='human', close='False'):
        # TODO: Plot embodiments/update plot
        pass

    def _joint_state_callback(self, joint_state):
        """
        Callback function for the joint_state_subscriber. Saves the received position and velocity.
        :param joint_state: Received data from joint_states topic.
        """
        # if DEBUG: print("Callback start")
        if np.isnan(joint_state.position).any() or np.isnan(joint_state.velocity).any():
            if DEBUG: print("Received NaN in learner data!")
            return
        self.l_position = joint_state.position
        self.l_velocity = joint_state.velocity
        # if DEBUG: print("Callback end")
        if not self._received_first_ldata:
            self._received_first_ldata = True

    def _restart_joint_state_controller(self):
        """
        It is necessary to restart the joint state controller each time after unpausing gazebo in order to publish/
        receive joint_state messages.
        :return: None
        """
        self._switch_controller_service(stop_controllers=['franka_sim_state_controller'],
                                        strictness=SwitchControllerRequest.BEST_EFFORT)
        self._switch_controller_service(start_controllers=['franka_sim_state_controller'],
                                        strictness=SwitchControllerRequest.BEST_EFFORT)

    def _get_expert_state_from_bagfile(self, step):
        """
        Gets a joint_message to the corresponding time of the given step from the bagfile and extracts the joint
        position and velocity (representing the state of the expert).
        :param step: Current step number.
        :return: Lists containing the joint positions and velocities
        """
        if DEBUG: print("Getting expert state from bagfile.")
        # Go one step ahead, otherwise data from bag will have a little delay
        step += 1
        expert_joint_state = next(self.bagfile.read_messages(topics=['/panda1/joint_states'],
                                                             start_time=self.bagfile_start_time + rospy.Duration(self.step_size * step),
                                                             end_time=self.bagfile_start_time + rospy.Duration(self.duration)))[1]
        return expert_joint_state.position, expert_joint_state.velocity

    def _calculate_reward(self, e_angles, e_angle_velocities, l_angles, l_angle_velocities):
        e_data_matrices, e_absolute_joint_frames = self.e_embodiment.data_matrices(e_angles, e_angle_velocities)
        l_data_matrices, l_absolute_joint_frames = self.l_embodiment.data_matrices(l_angles, l_angle_velocities)

        data_matrix_combinations = cartesian_product(e_data_matrices, l_data_matrices, flat=True)
        distances = [-link_distance(e_link, l_link) for e_link, l_link in data_matrix_combinations]
        distance_matrix = np.reshape(distances, [self.e_embodiment.num_links, self.l_embodiment.num_links])
        weighted_matrix = np.multiply(distance_matrix, self.weight_matrix)
        reward = np.mean(weighted_matrix)
        return reward


if __name__ == '__main__':
    env = GazeboEnvFullPanda(0.1, 4.0, '../resources/torque_trajectory_002.bag', example_embodiments.panda_embodiment, example_embodiments.panda_embodiment)
    env.reset()
    #quit()

    with open('../resources/torque_trajectory_002_commands.csv', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for line in csv_reader:
            obs, reward, done, _ = env.step(line)
            if done: break
            #print(line)
            #print(env._get_expert_state_from_bagfile(env.current_step)[2])
            print("")

    # with rosbag.Bag('../resources/torque_trajectory_002.bag') as bagfile:
    #     start_time = rospy.Time(bagfile.get_start_time())
    #     for i in range(130):
    #         print(next(bagfile.read_messages(topics=['/panda1/joint_states'],
    #                                          start_time=start_time + rospy.Duration(env.step_size * i))))
