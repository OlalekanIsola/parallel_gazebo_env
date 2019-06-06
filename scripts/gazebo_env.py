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


class GazeboEnvFullPanda(gym.Env):
    """
    An openAI gym environment to learn how to copy a robot motion.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size, bagfile):
        super(GazeboEnvFullPanda, self).__init__()
        self.current_step = 0
        self.step_size = step_size
        self._received_first_ldata = False
        self.l_position = [0.0] * 7
        self.l_velocity = [0.0] * 7
        self.e_position = [0.0] * 7
        self.e_velocity = [0.0] * 7

        self.e_effort = [0.0] * 7

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
        self._command_zero.data = [0.0] * 7

        self._pause_gazebo_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_gazebo_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset_gazebo_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._switch_controller_service = rospy.ServiceProxy('panda1/controller_manager/switch_controller', SwitchController)
        if DEBUG: print("Services registered.")

        # TODO: Find good value for max reward
        self.reward_range = (-(4*174 + 72), 0)

        # Joint torque ranges
        self.action_space = gym.spaces.Box(
            low=np.array([-87, -87, -87, -87, -12, -12, -12]),
            high=np.array([87, 87, 87, 87, 12, 12, 12]),
            dtype='float32')

        # Respectively: l_joint_angles, l_joint_vels, t_joint_angles, t_joint_vels
        self.observation_space = gym.spaces.Box(
            low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
            dtype='float32')

        # self.observation_space = gym.spaces.Box(
        #     low=np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        #                   [-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100],
        #                   [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        #                   [-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]]),
        #     high=np.array([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        #                    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
        #                    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        #                    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]]),
        #     dtype='float32')

    def __del__(self):
        if DEBUG: print("Destructor.")
        self.bagfile.close()

    def reset(self):
        if DEBUG: print("Reset method start")
        self._command_publisher.publish(self._command_zero)
        if DEBUG: print("Zero command published")
        self._reset_gazebo_service()
        if DEBUG: print("Gazebo reset.")
        self._unpause_gazebo_service()
        if DEBUG: print("Gazebo unpaused.")
        self._restart_joint_state_controller()
        if DEBUG: print("Joint State Controller resetted.")
        self._received_first_ldata = False
        while self._received_first_ldata is False:
            pass
        self._pause_gazebo_service()
        if DEBUG: print("Gazebo paused")
        self.current_step = 0
        if DEBUG: print("Reset method end.")
        #return [self.l_position, self.l_velocity, self.e_position, self.e_velocity]
        return self.e_position

    def step(self, action):
        assert len(action) is 7, "Action needs to consist of 7 numbers!"
        # self._unpause_gazebo_service()
        # self._restart_joint_state_controller()
        # self._command.data = action
        # self._command_publisher.publish(self._command)
        # rospy.sleep(self.step_size)
        # self._pause_gazebo_service()
        self.current_step += 1
        # TODO Last observation
        # TODO Calculate reward
        done = False

        # For testing: Just compare taken action with action from teacher ~behavioral cloning
        try:
            self.e_position, self.e_velocity, self.e_effort = self._get_expert_state_from_bagfile(self.current_step-1)
        except StopIteration:
            done = True

        reward = -np.linalg.norm(np.array(action) - np.array(self.e_effort))
        #observation = [self.l_position, self.l_velocity, self.e_position, self.e_velocity]
        observation = self.e_position

        # if DEBUG: print("[{}] Comparing (Reward: {})".format(self.current_step, reward))
        # if DEBUG: print(action)
        # if DEBUG: print(self.e_effort)
        # if DEBUG: print("")
        print("[{}] Comparing (Reward: {})".format(self.current_step, reward))
        print(action)
        print(self.e_effort)
        print("")

        return observation, reward, done, {}

    def render(self, mode='human', close='False'):
        pass

    def _joint_state_callback(self, joint_state):
        """
        Callback function for the joint_state_subscriber. Saves the received position and velocity.
        :param joint_state: Received data from joint_states topic.
        """
        # if DEBUG: print("Callback start")
        self.l_position = joint_state.position
        self.l_velocity = joint_state.velocity
        # if DEBUG: print("Callback end")
        if not self._received_first_ldata:
            self._received_first_ldata = True

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

    def _get_expert_state_from_bagfile(self, step):
        if DEBUG: print("Getting expert state from bagfile.")
        # Go one step ahead, otherwise data from bag will have a little delay
        step += 1
        expert_joint_state = next(self.bagfile.read_messages(topics=['/panda1/joint_states'],
                                                             start_time=self.bagfile_start_time + rospy.Duration(self.step_size * step),
                                                             end_time=self.bagfile_start_time + rospy.Duration(10.0)))[1]
        return expert_joint_state.position, expert_joint_state.velocity, expert_joint_state.effort


if __name__ == '__main__':
    env = GazeboEnvFullPanda(0.1, '../resources/torque_trajectory_002.bag')
    env.reset()
    if DEBUG: print("We end here.")
    # quit()

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
