#!/usr/bin/env bash

export ROS_MASTER_URI=http://localhost:11350
export GAZEBO_MASTER_URI=http://localhost:11340

roslaunch franka_gazebo franka_gazebo --screen
