from nature import Agent
import optuna


optuna.create_study().optimize(lambda hp: Agent(hp).objective)
