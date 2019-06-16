#!/usr/bin/env python3

from gazebo_env import GazeboEnvFullPanda
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import  MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = DummyVecEnv([lambda: GazeboEnvFullPanda(0.1, '../resources/torque_trajectory_002.bag', 10.0)])
model = PPO2.load("ppo2_identical_pandas_copy_torques", tensorboard_log="./ppo2_identical_pandas_copy_torques_tensorboard/")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)