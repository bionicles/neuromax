# neuromax.py - why?: solve PyMolEnv-v0 with RL
# spinningup.openai.com/en/latest/user/running.html#launching-from-scripts
# note: we may need to hack gym source code to register our env in all threads
# /home/bion/anaconda3/envs/tfg/lib/python3.6/site-packages/gym

from spinup.utils.run_utils import ExperimentGrid
import tensorflow as tf
from spinup import td3


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='neuromax')
    eg.add('env_name', 'PyMolEnv-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add(
        'ac_kwargs:hidden_sizes',
        [(18, 18)],
        'hid')
    eg.add('ac_kwargs:activation', [tf.tanh], '')
    eg.run(td3, num_cpu=args.cpu)
