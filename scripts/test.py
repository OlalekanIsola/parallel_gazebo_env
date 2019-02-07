#! /usr/bin/env python

import os
import time
import copy
import multiprocessing as mp

import rospy
from tf.transformations import euler_from_quaternion
from controller_manager_msgs.srv import SwitchController, LoadController
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

#adjust as much as you need, limited number of physical cores of your cpu
cpu_processes = 2


class Worker(mp.Process):
    def __init__(self, someGlobalNumber, id, limit):
        super(Worker, self).__init__()
        self.port = 11350 + id
        self.global_number = someGlobalNumber
        self.limit = limit


    def run(self):
        #to parallelizing
        os.environ['ROS_MASTER_URI'] = "http://localhost:" + str(self.port) + '/'
        rospy.init_node('parallelSimulationNode')
        rospy.wait_for_service('/panda1/controller_manager/switch_controller')
        try:
            load_controller = rospy.ServiceProxy('/panda1/controller_manager/load_controller', LoadController)
            switch_controller = rospy.ServiceProxy('/panda1/controller_manager/switch_controller', SwitchController)
            result = load_controller('position_joint_trajectory_controller')
            result = switch_controller(['position_joint_trajectory_controller'], [], 2)
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e

        pub = rospy.Publisher('/panda1/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)

        joint_trajectory = JointTrajectory(joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
        joint_trajectory.points = [JointTrajectoryPoint()]
        joint_trajectory.points[0].positions = [self.limit, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_trajectory.points[0].velocities = [0.0 for i in range(0, 7)]
        joint_trajectory.points[0].accelerations = [0.0 for i in range(0, 7)]
        joint_trajectory.points[0].effort = [0.0 for i in range(0, 7)]
        joint_trajectory.points[0].time_from_start = rospy.Duration(1.0)

        joint_trajectory2 = copy.deepcopy(joint_trajectory)
        joint_trajectory2.points[0].positions[0] = -self.limit

        rate = rospy.Rate(0.2)
        forward = True
        while not rospy.is_shutdown():
            if forward is True:
                #joint_trajectory.header.stamp = rospy.Time.now()
                pub.publish(joint_trajectory)
                forward = False
            else:
                pub.publish(joint_trajectory2)
                forward = True
            with self.global_number.get_lock():
                self.global_number.value += 1
            print self.port, ' global_number: ', self.global_number.value
            rate.sleep()


if __name__ == "__main__":
    #initializing some shared values between process
    global_number = mp.Value('i', 0)

    # parallel training
    #workers = [Worker(global_number, i) for i in range(cpu_processes)]
    workers = [Worker(global_number, 0, 1.0), Worker(global_number, 1, 0.1)]
    [w.start() for w in workers]
    [w.join() for w in workers]
