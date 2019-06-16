#!/usr/bin/env python3

from gazebo_env import GazeboEnvFullPanda
import gym
import example_embodiments
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv


n_cpu = 1
env = DummyVecEnv([lambda: GazeboEnvFullPanda(0.1, 4.0, '../resources/torque_trajectory_002.bag', example_embodiments.panda_embodiment, example_embodiments.panda_embodiment)])
model = PPO2(MlpLstmPolicy, env, verbose=2,
             tensorboard_log="./ppo2_identical_pandas_tensorboard_2019_06_16/",
             gamma=0.1,
             # n_steps=10,
             nminibatches=1
             )
model.learn(total_timesteps=10000000)
model.save("ppo2_identical_pandas_copy_torques_2019_06_16")