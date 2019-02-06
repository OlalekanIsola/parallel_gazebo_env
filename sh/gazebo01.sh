#!/usr/bin/env bash

export ROS_MASTER_URI=http://localhost:11351
export GAZEBO_MASTER_URI=http://localhost:11341

roslaunch franka_gazebo franka_gazebo --screen
