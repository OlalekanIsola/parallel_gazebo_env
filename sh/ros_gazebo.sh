#!/usr/bin/env bash

offset=$1
rosPort=$((11350 + offset))
gazeboPort=$((11330 + offset))

export ROS_MASTER_URI=http://localhost:$rosPort
export GAZEBO_MASTER_URI=http://localhost:$gazeboPort

roslaunch franka_gazebo franka_gazebo.launch --screen spawn_controller:=true
