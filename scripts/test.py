#! /usr/bin/env python

import os
import multiprocessing as mp

import rospy
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from controller_manager_msgs.srv import SwitchController
#from geometry_msgs.msg import
from gazebo_msgs.msg import ModelState
import time

#adjust as much as you need, limited number of physical cores of your cpu
cpu_processes = 2

#######ROS PARAMETERS############
x = 0.0
y = 0.0
x_vel = 0.0
y_vel = 0.0
theta = 0.0
speed = Twist()


class Worker(mp.Process):
    def __init__(self, someGlobalNumber, name, limit):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.port = name
        self.global_number = someGlobalNumber
        self.limit = limit


    def run(self):
        #to parallelizing
        os.environ['ROS_MASTER_URI'] = "http://localhost:1135" + str(self.port) + '/'
        rospy.init_node('parallelSimulationNode')
        rospy.wait_for_service('/panda1/controller_manager/switch_controller')
        #load controller first!
        try:
            switch_controller = rospy.ServiceProxy('/panda1/controller_manager/switch_controller', SwitchController)
            result = switch_controller(['position_joint_trajectory_controller'], [], 2)
        except: rospy.ServiceException, e:
            print "Service call failed: %s" %e


        #pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)

        #rate = rospy.Rate(10)
        #while not rospy.is_shutdown():
        #    speed.linear.x = 0.4
        #    speed.angular.z = 0.5
        #    if self.port % 2 == 0:
        #        speed.angular.z = -1.5

            #Publishing speed Values to generate new states#######################################
        #    pub.publish(speed)
        #    with self.global_number.get_lock():
        #        self.global_number.value += 1
        #    print self.name, 'global_number: ', self.global_number.value
        #    rate.sleep()








if __name__ == "__main__":
    #initializing some shared values between process
    global_number = mp.Value('i', 0)

    # parallel training
    #workers = [Worker(global_number, i) for i in range(cpu_processes)]
    workers = [Worker(global_number, 0, 1.0), Worker(global_number, 0, 0.1)]
    [w.start() for w in workers]
    [w.join() for w in workers]
