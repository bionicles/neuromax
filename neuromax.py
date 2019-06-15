#NeuroImpala, why? use neuromax with IMPALA
from mol import PyMolEnv
import ray
from resnet import ResNet
from ray.tune.registry import register_env
from ray.rllib.agents import impala # https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/impala/impala.py

neuromax_env = PyMolEnv()

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    register_env("neuromax", PyMolEnv)
    ray.init()
    trainer = impala.ImpalaTrainer(env="neuromax_env")
