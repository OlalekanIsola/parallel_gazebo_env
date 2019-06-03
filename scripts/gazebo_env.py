#!/usr/bin/env python3

# import sys
# import copy
import rospy
import gym
import csv
import time
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest


class GazeboEnvFullPanda(gym.Env):
    """
    An openAI gym environment to learn how to copy a robot motion.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size):
        super(GazeboEnvFullPanda, self).__init__()
        self.step_size = step_size
        self.l_positions = [0.0] * 7
        self.l_velocities = [0.0] * 7
        self.t_positions = [0.0] * 7
        self.t_velocities = [0.0] * 7

        rospy.init_node('gym_environment_wrapper')

        self._joint_states_subscriber = rospy.Subscriber('panda1/joint_states', JointState, self._joint_state_callback, queue_size=1)

        self._command_publisher = rospy.Publisher('panda1/effort_jointgroup_controller/command', Float64MultiArray, queue_size=1)
        self._command = Float64MultiArray()
        self._command_zero = Float64MultiArray()
        self._command_zero.data = [0.0] * 7

        self._pause_gazebo_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_gazebo_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset_gazebo_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._switch_controller_service = rospy.ServiceProxy('panda1/controller_manager/switch_controller', SwitchController)

        # TODO: Find good value for max reward
        self.reward_range = (-1000, 0)

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

    def reset(self):
        self._command_publisher.publish(self._command_zero)
        self._reset_gazebo_service()
        self._pause_gazebo_service()

    def step(self, action):
        assert len(action) is 7, "Action needs to consist of 7 numbers!"
        self._unpause_gazebo_service()
        self._restart_joint_state_controller()
        self._command.data = action
        self._command_publisher.publish(self._command)
        rospy.sleep(self.step_size)
        # TODO Last observation
        self._pause_gazebo_service()
        # TODO Calculate reward

    def render(self, mode='human', close='False'):
        pass

    def _joint_state_callback(self, joint_state):
        self.l_positions = joint_state.position
        self.l_velocities = joint_state.velocity

    def _restart_joint_state_controller(self):
        """
        It is necessary to restart the joint state controller each time after unpausing gazebo in order to publish/
        receive joint_state messages.
        :return: none
        """
        self._switch_controller_service(stop_controllers=['franka_sim_state_controller'],
                                        strictness=SwitchControllerRequest.BEST_EFFORT)
        self._switch_controller_service(start_controllers=['franka_sim_state_controller'],
                                        strictness=SwitchControllerRequest.BEST_EFFORT)


if __name__ == '__main__':
    env = GazeboEnvFullPanda(0.1)
    env.reset()

    with open('../resources/torque_trajectory_002_commands.csv', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for line in csv_reader:
            env.step(line)
