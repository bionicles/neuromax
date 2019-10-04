from tools import get_timestamp
from tabulate import tabulate
from nature import Agent
import optuna

RESUME_FROM_DB = True
DB_NAME = 'neuromax'
N_TRIALS = 2  # None  # to run continually


def run_study():
    study = optuna.create_study(
        study_name=get_timestamp(), storage=f'sqlite:///{DB_NAME}.db',
        load_if_exists=RESUME_FROM_DB)
    study.optimize(lambda trial: Agent(trial).objective, n_trials=N_TRIALS)

    optuna.visualization.plot_intermediate_values(study)
    print(tabulate(study.trials_dataframe()))
    return study


if __name__ is "__main__":
    run_study()
