#! /usr/bin/env python3

import time
import numpy as np
import rospy
from std_srvs.srv import Empty
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from std_msgs.msg import Float64MultiArray
import subprocess
import os
import shlex

lower_jointpos_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, 0.1])
upper_jointpos_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 0.1])
joint_pos_interval = upper_jointpos_limits - lower_jointpos_limits
def random_angles():
    return np.random.rand(7) * joint_pos_interval + lower_jointpos_limits

rospy.init_node('gym_environment_wrapper')

command_publisher = rospy.Publisher('panda1/effort_position_jointgroup_controller/command', Float64MultiArray, queue_size=1)
pause_gazebo_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
unpause_gazebo_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
reset_gazebo_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
switch_controller_service = rospy.ServiceProxy('panda1/controller_manager/switch_controller', SwitchController)
command = Float64MultiArray()

for i in range(4, 100):
    reset_gazebo_service()
    unpause_gazebo_service()
    switch_controller_service(stop_controllers=["franka_sim_state_controller"],
                              strictness=SwitchControllerRequest.BEST_EFFORT)
    switch_controller_service(start_controllers=["franka_sim_state_controller"],
                              strictness=SwitchControllerRequest.BEST_EFFORT)
    command.data = [0.0] * 7
    command_publisher.publish(command)
    rosbag_command = shlex.split("rosbag record -O ../resources/torque_trajectory_{0:03}.bag /panda1/joint_states __name:=my_bag".format(i))
    rosbag_proc = subprocess.Popen(rosbag_command)
    time.sleep(0.5)
    command.data = random_angles().tolist()
    command_publisher.publish(command)
    time.sleep(7)
    os.system('rosnode kill /my_bag')

pause_gazebo_service()