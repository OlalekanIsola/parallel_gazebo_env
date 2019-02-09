#! /usr/bin/env python

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction

class ILEnvironment(object):
    def __init__(self):
        super(ILEnvironment, self).__init__()

        self.joint_position_action_client = actionlib.SimpleActionClient("position_joint_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        while not rospy.is_shutdown() and self.joint_position_action_client.wait_for_server(rospy.Duration(4.0)) is False:
            rospy.logwarn("Waiting for follow_joint_trajectory action server...")

    def reset_environment(self):
        pass

    def get_state(self):
        pass

    def step(self):
        pass
