import optuna

from nature import Agent
from tools import log


def objective(hp):
    return Agent(hp).objective


study = optuna.create_study()
study.optimize(objective, n_trials=10)
log(study.best_params)
