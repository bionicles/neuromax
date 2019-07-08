import tensorflow as tf
from ray import tune
import subprocess
import time
import ray
import os

# open a new terminal and monitor nvidia
# os.system("gnome-terminal -e 'watch -n0.1 nvidia-smi'")

# test gpu
# you need to have mesa-utils installed for this to work
process = subprocess.Popen(["glxgears"])
time.sleep(6)
process.kill()

# test tensorflow-gpu
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# test ray
ray.init(num_gpus=1)
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,
        "num_workers": 11,
        "lr": tune.grid_search([0.01, 0.001])
    },
)
tune.run(
    "APEX",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,
        "num_workers": 11,
        "lr": tune.grid_search([0.01, 0.001])
    },
)
tune.run(
    "APPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,
        "num_workers": 11,
        "lr": tune.grid_search([0.01, 0.001])
    },
)
tune.run(
    "IMPALA",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,
        "num_workers": 11,
        "lr": tune.grid_search([0.01, 0.001])
    },
)
