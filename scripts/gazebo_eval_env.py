#!/usr/bin/env python3

import gazebo_env
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest


class GazeboEvalEnv(gazebo_env.GazeboEnv):
    """
    An environment to evaluate an agent trained on GazeboEnv. The step function of this environment sends commands to
    two robots to show them moving next to each other.
    """
    def __init__(self, step_size, duration, bagfiles, e_embodiment, l_embodiment):
        super(GazeboEvalEnv, self).__init__(step_size, duration, bagfiles, e_embodiment, l_embodiment)

        self._expert_command = Float64MultiArray()
        self._expert_command.data = [0.0] * e_embodiment.num_links

        self._expert_command_publisher = rospy.Publisher('panda2/effort_jointgroup_controller/command', Float64MultiArray,
                                                         queue_size=1)
        self._switch_controller2_service = rospy.ServiceProxy('panda2/controller_manager/switch_controller',
                                                              SwitchController)

    # def reset(self):
    #     super().reset()
    #     self._restart_controller2('franka_sim_state_controller')
    #     print("Restarting second controller")

    def step(self, action):
        """
        Executes the given action by publishing the joint effort command to the ROS controller. In order to receive the
        joint_states messages properly, the joint_state_controller needs to be restarted.
        :param action: A list of joint efforts to send to the joints of the learner.
        :return: The current state/observation, the immediate reward and the 'done' flag.
        """
        done = False
        try:
            self.e_position, self.e_velocity, self.e_effort = self._get_expert_state_from_bagfile(self.last_time_stamp)
        except StopIteration:
            done = True

        self._unpause_gazebo_service()
        self._restart_state_controller()
        self._command.data = action
        self._expert_command.data = self.e_effort
        self._command_publisher.publish(self._command)
        self._expert_command_publisher.publish(self._expert_command)
        try:
            rospy.sleep(self.step_size)
            slept = True
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            slept = False
        if slept is False:
            rospy.sleep(self.step_size)
        self._pause_gazebo_service()
        self.current_step += 1

        reward = self._calculate_reward(self.e_position, self.e_velocity, self.l_position, self.e_velocity)
        observation = np.concatenate([self.l_position, self.l_velocity, self.e_position, self.e_velocity])

        return observation, reward, done, {}

    def _restart_state_controller(self):
        """
        It is necessary to restart the joint state controller each time after unpausing gazebo in order to publish/
        receive joint_state messages.
        :return: None
        """
        super()._restart_state_controller()
        self._switch_controller2_service(stop_controllers=['franka_sim_state_controller'],
                                         strictness=SwitchControllerRequest.BEST_EFFORT)
        self._switch_controller2_service(start_controllers=['franka_sim_state_controller'],
                                         strictness=SwitchControllerRequest.BEST_EFFORT)
