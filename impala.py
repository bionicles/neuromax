import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

if __name__ == "__main__":
    ray.init()

    pbounds = {
        'GAIN': (1e-4, 0.1),
        'UNITS': (17, 2000),
        'LR': (1e-4, 1e-1),
        'EPSILON': (1e-4, 1),
        'LAYERS': (1, 10),
        'BLOCKS': (1, 50),
        'L1': (0, 0.1),
        'L2': (0, 0.1),
    }
