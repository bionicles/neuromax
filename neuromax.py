from tools import get_timestamp, log
from tabulate import tabulate
from nature import Agent
import optuna

STUDY_NAME = get_timestamp()
RESUME_FROM_DB = True
DB_NAME = 'neuromax'
N_TRIALS = None  # use None to run continually


def objective(trial):
    agent = Agent(trial)
    objective = agent.objective
    del agent
    return objective


def run_study():
    log('BEGIN STUDY:', STUDY_NAME, color="blue")
    study = optuna.create_study(
        # comment out the next line to use in-memory storage (not persistent)
        storage=f'sqlite:///{DB_NAME}.db', load_if_exists=RESUME_FROM_DB,
        study_name=STUDY_NAME)
    study.optimize(objective, n_trials=N_TRIALS)

    print(tabulate(study.trials_dataframe()))
    return study


run_study()
