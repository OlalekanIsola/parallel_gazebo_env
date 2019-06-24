#!/usr/bin/env python3

from gazebo_env import GazeboEnv
from gazebo_eval_env import GazeboEvalEnv
import example_embodiments
from stable_baselines import PPO2
from stable_baselines.common.policies import  MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = DummyVecEnv([lambda: GazeboEvalEnv(0.1, 5.0, '../resources/torque_trajectory_003.bag', example_embodiments.panda_embodiment, example_embodiments.panda_embodiment)])
model = PPO2.load("../runs/models/e7_l7_4sec_gamma0.4_callbacktest_best", tensorboard_log="../runs/tensorboard/ppo2_identical_pandas_2019_06_18/")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)