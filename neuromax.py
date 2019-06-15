#NeuroImpala, why? use neuromax with IMPALA
from mol import PyMolEnv
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import impala

neuromax_env = PyMolEnv()

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    register_env("neuromax", PyMolEnv)
    ray.init()
    trainer = ppo.PPOTrainer(env="neuromax")
