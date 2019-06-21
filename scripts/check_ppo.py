#!/usr/bin/env python3

from gazebo_env import GazeboEnvFullPanda
import example_embodiments
from stable_baselines import PPO2
from stable_baselines.common.policies import  MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = DummyVecEnv([lambda: GazeboEnvFullPanda(0.1, 3.0, '../resources/torque_trajectory_002.bag', example_embodiments.panda_embodiment, example_embodiments.panda_embodiment)])
model = PPO2.load("../runs/models/ppo2_identical_pandas_4s_2019_06_19", tensorboard_log="../runs/tensorboard/ppo2_identical_pandas_2019_06_18/")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)