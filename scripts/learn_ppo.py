#!/usr/bin/env python3

from gazebo_env import GazeboEnv
import time
import numpy as np
import example_embodiments
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import ts2xy, load_results
from stable_baselines.bench import Monitor

description_str = "trajectory100_e7_l3_5sec_gamma0.4_t0_r1.0_lin0.001_rot0.01"
log_dir = "../runs/tensorboard/ppo2_2019_07_03_trajectory002/"
continue_learning = False
# bagfiles = ["../resources/torque_trajectory_{0:03}.bag".format(i) for i in range(2, 10)]
bagfiles = ["../resources/torque_trajectory_100.bag"]
gazebo_env = GazeboEnv(0.1, 5.0, bagfiles, example_embodiments.panda_embodiment, example_embodiments.panda_3j_embodiment)
env = DummyVecEnv([lambda: Monitor(gazebo_env, "../runs/monitor/", allow_early_resets=True)])

if continue_learning:
    model = PPO2.load("../runs/models/trajectory003_e7_l4_5sec_gamma0.3_t0_r1.0_lin0.001_rot0.01_best.pkl",
                      env=env,
                      tensorboard_log=log_dir)
else:
    model = PPO2(MlpPolicy, env, verbose=2,
                 tensorboard_log=log_dir,
                 gamma=0.4,
                 # n_steps=30,
                 # nminibatches=1
                 )

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 256 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results("../runs/monitor/"), 'timesteps')
        if len(x) > 0:
            start_time = time.time()
            mean_reward = np.mean(y[-12800:])
            print("{} seconds for mean calculation".format(time.time() - start_time))
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f}, Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save("../runs/models/" + description_str + "_best.pkl")
    n_steps += 1
    return True


model.learn(total_timesteps=10000000,
            tb_log_name=description_str,
            callback=callback)

answer = input("Save Model? (Y/n)")
if answer == "n" or answer == "N":
    pass
else:
    name = input("Enter filename:")
    model.save("../runs/models/" + description_str + "_final.pkl")
