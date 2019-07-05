#!/usr/bin/env python3

from gazebo_env import GazeboEnv
from gazebo_eval_env import GazeboEvalEnv
import example_embodiments
from stable_baselines import PPO2
from stable_baselines.common.policies import  MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# bagfiles = ["../resources/torque_trajectory_{0:03}.bag".format(i) for i in range(2, 100)]
bagfiles = ["../resources/torque_trajectory_002.bag"]
env = DummyVecEnv([lambda: GazeboEvalEnv(0.1, 5.0, bagfiles, example_embodiments.panda_embodiment, example_embodiments.panda_4j_embodiment)])
model = PPO2.load("../runs/models/trajectory002_e7_l4_5sec_gamma0.3_t0_r1.0_lin0.001_rot0.01_best", tensorboard_log="../runs/tensorboard/ppo2_2019_07_03_trajectory002/")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)