#!/usr/bin/env python3

# import sys
# import copy
import rospy
import rosbag
import gym
import csv
import time
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest


DEBUG = False
DEBUG2 = False


class GazeboEnvFullPanda(gym.Env):
    """
    An openAI gym environment to learn how to copy a robot motion.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size, bagfile):
        super(GazeboEnvFullPanda, self).__init__()
        self.current_step = 0
        self.step_size = step_size

        self.e_position = [0.0] * 7
        self.e_velocity = [0.0] * 7
        self.e_effort = [0.0] * 7

        self.bagfile = rosbag.Bag(bagfile)
        self.bagfile_start_time = rospy.Time(self.bagfile.get_start_time())
        if DEBUG: print("Bagfile opened.")


        # TODO: Find good value for max reward
        self.reward_range = (-(4*174 + 72), 0)

        # Joint torque ranges
        self.action_space = gym.spaces.Box(
            low=np.array([-87, -87, -87, -87, -12, -12, -12]),
            high=np.array([87, 87, 87, 87, 12, 12, 12]),
            dtype='float32')

        # Respectively: t_joint_angles
        self.observation_space = gym.spaces.Box(
            low=np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                         [-87, -87, -87, -87, -12, -12, -12]]),
            high=np.array([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                          [87, 87, 87, 87, 12, 12, 12]]),
            dtype='float32')

    def __del__(self):
        if DEBUG: print("Destructor.")
        self.bagfile.close()

    def reset(self):
        if DEBUG: print("Reset method start")
        self.current_step = 0
        self.e_position, self.e_velocity, self.e_effort = self._get_expert_state_from_bagfile(self.current_step)
        if DEBUG: print("Reset method end.")
        return [self.e_position, self.e_effort]

    def step(self, action):
        self.current_step += 1
        done = False

        # For testing: Just compare taken action with action from teacher ~behavioral cloning
        try:
            self.e_position, self.e_velocity, self.e_effort = self._get_expert_state_from_bagfile(self.current_step-1)
        except StopIteration:
            done = True

        observation = [self.e_position, self.e_effort]
        reward = -np.linalg.norm(np.array(action) - np.array(self.e_effort))

        if DEBUG2: print("[{}] Comparing (Reward: {})".format(self.current_step, reward))
        if DEBUG2: print(action)
        if DEBUG2: print(self.e_effort)
        if DEBUG2: print("")

        return observation, reward, done, {}

    def render(self, mode='human', close='False'):
        pass

    def _get_expert_state_from_bagfile(self, step):
        if DEBUG: print("Getting expert state from bagfile.")
        # Go one step ahead, otherwise data from bag will have a little delay
        step += 1
        expert_joint_state = next(self.bagfile.read_messages(topics=['/panda1/joint_states'],
                                                             start_time=self.bagfile_start_time + rospy.Duration(self.step_size * step),
                                                             end_time=self.bagfile_start_time + rospy.Duration(10.0)))[1]
        return expert_joint_state.position, expert_joint_state.velocity, expert_joint_state.effort


if __name__ == '__main__':
    env = GazeboEnvFullPanda(1.0, '../resources/torque_trajectory_002.bag')
    env.reset()
    if DEBUG: print("We end here.")
    # quit()
    cumreward = 0
    with open('../resources/torque_trajectory_002_commands.csv', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for line in csv_reader:
            obs, reward, done, _ = env.step(line)
            cumreward += reward
            if done: break
            #print(line)
            #print(env._get_expert_state_from_bagfile(env.current_step)[2])
            print("")
    print("\nCumulative Reward: {}".format(cumreward))

    # with rosbag.Bag('../resources/torque_trajectory_002.bag') as bagfile:
    #     start_time = rospy.Time(bagfile.get_start_time())
    #     for i in range(130):
    #         print(next(bagfile.read_messages(topics=['/panda1/joint_states'],
    #                                          start_time=start_time + rospy.Duration(env.step_size * i))))
