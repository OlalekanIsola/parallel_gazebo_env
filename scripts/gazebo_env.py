#!/usr/bin/env python3

# import sys
# import copy
import rospy
import gym
import time
import numpy as np
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty


class GazeboEnvFullPanda(gym.Env):
    """
    An openAI gym environment to learn how to copy a robot motion.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GazeboEnvFullPanda, self).__init__()

        rospy.init_node('gym_environment_wrapper')

        self._command_publisher = rospy.Publisher('panda1/effort_jointgroup_controller/command', Float64MultiArray, queue_size=1)
        self._command = Float64MultiArray()
        self._command_zero = Float64MultiArray()
        self._command_zero.data = [0.0] * 7

        self._pause_gazebo_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_gazebo_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset_gazebo_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

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
        pass

    def step(self, action):
        # Call gazebo service to unpause physics
        # Send effort command to controller
        # Run 0.Xs (get sim/ROS time)
        # Save last observation
        # Pause
        # Calculate reward
        pass

    def render(self, mode='human', close='False'):
        pass
