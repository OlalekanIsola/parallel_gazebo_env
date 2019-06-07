#!/usr/bin/env python3

from gazebo_env import GazeboEnvFullPanda
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import  MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


n_cpu = 1
env = DummyVecEnv([lambda: GazeboEnvFullPanda(0.1, '../resources/torque_trajectory_002.bag')])
model = PPO2(MlpPolicy, env, verbose=2,
             tensorboard_log="./ppo2_identical_pandas_copy_torques_tensorboard/",
             gamma=0.0,
             #n_steps=10,
             #nminibatches=5
             )
model.learn(total_timesteps=1000000)
model.save("ppo2_identical_pandas_copy_torques")