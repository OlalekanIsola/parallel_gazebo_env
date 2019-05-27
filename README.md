# Parallel Gazebo openAI environment

To start instances of Gazebo run
```
sh sh/ros_gazebo.sh [instanceID]
```

To make use of openAI baselines, this "package" uses python3. Some rospy calls work from within python3 but the scripts of course need to be run directly instead of calling rosrun or roslaunch.
