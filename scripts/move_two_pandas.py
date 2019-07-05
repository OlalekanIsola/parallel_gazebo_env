#!/usr/bin/env python3

import sys
import copy
import rospy
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


def publisher():
    panda1_publisher = rospy.Publisher('panda1/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
    panda2_publisher = rospy.Publisher('panda2/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
    rospy.init_node('joint_pose_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    panda1_trajectory = JointTrajectory()
    panda1_trajectory.joint_names = ['panda1_joint1', 'panda1_joint2', 'panda1_joint3', 'panda1_joint4', 'panda1_joint5', 'panda1_joint6', 'panda1_joint7']
    panda1_trajectory.points.append(JointTrajectoryPoint())
    panda1_trajectory.points[0].time_from_start = rospy.Duration(1.0)
    panda1_trajectory.points[0].velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    panda1_trajectory.points[0].effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    panda1_trajectory.points[0].accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    panda2_trajectory = JointTrajectory()
    panda2_trajectory.joint_names = ['panda2_joint1', 'panda2_joint2', 'panda2_joint3', 'panda2_joint4', 'panda2_joint5', 'panda2_joint6', 'panda2_joint7']
    panda2_trajectory.points.append(JointTrajectoryPoint())
    panda2_trajectory.points[0].time_from_start = rospy.Duration(1.0)
    panda2_trajectory.points[0].velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    panda2_trajectory.points[0].effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    panda2_trajectory.points[0].accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    while not rospy.is_shutdown():
        panda1_trajectory.header = std_msgs.msg.Header()
        panda2_trajectory.header = std_msgs.msg.Header()

        panda1_trajectory.points[0].positions = [1.0, -1.0, 0.5, 0.5, 0.5, 0.0, 0.0]
        panda2_trajectory.points[0].positions = [1.0, -1.0, 0.5, 0.5, 0.5, 0.0, 0.0]

        panda1_publisher.publish(panda1_trajectory)
        panda2_publisher.publish(panda2_trajectory)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
