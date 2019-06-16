#!/usr/bin/env python3

from gazebo_env import GazeboEnvFullPanda
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import  MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import csv
import example_embodiments


env = GazeboEnvFullPanda(0.1, 4.0, '../resources/torque_trajectory_002.bag', example_embodiments.panda_embodiment, example_embodiments.panda_embodiment)

env.reset()

with open('../resources/fail_commands_try.csv', newline='\n') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for line in csv_reader:
        obs, reward, done, _ = env.step(line)
        if done: break
        # print(line)
        # print(env._get_expert_state_from_bagfile(env.current_step)[2])
        print("")