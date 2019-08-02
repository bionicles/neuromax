# experiment.py: why? simplify
from plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
from tensorboard import program
from attrdict import AttrDict
import webbrowser
import datetime
import skopt
import time
import os

from nature.agent import Agent
from nurture.nurture import train
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=true'
import tensorflow as tf
# tracker = SummaryTracker()
plt.set_cmap("viridis")
B, L, K = tf.keras.backend, tf.keras.layers, tf.keras
# globals
proteins = None
trial_number = 0
atom_index = 0
best_trial = ''
best_args = []
best = 12345678900
# search
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
EPISODES_PER_TASK = 100
N_RANDOM_STARTS = 10
N_CALLS = 1000
# episodes
STOP_LOSS_MULTIPLE = 1.2
TEMPERATURE = 273.15
TENSORBOARD = False
MAX_STEPS = 420
# gif parameters
N_MOVIES = 2
# agent
WORD_VECTOR_LENGTH = 512
CODE = AttrDict({"shape": (None, 10), "dtype": tf.float32})
# tasks
tasks = [
        # {
        #     "type": "env",
        #     "name": "MountainCar-v0"
        # },
        {
            "type": "dataset",
            "name": "mol",
            "inputs": [AttrDict({"shape": (None, 10), "dtype": tf.float32})],
            "outputs": [AttrDict({"shape": (None, 3), "dtype": tf.float32})]
        },
        # {
        #     "type": "dataset",
        #     "name": "clevr",
        #     "inputs": [
        #         AttrDict({"shape": (480, 320, 3), "dtype": tf.float32}),
        #         AttrDict({"shape": (None, WORD_VECTOR_LENGTH), "dtype": tf.float32}),
        #                ],
        #     "outputs": [AttrDict({"shape": (28, ), "dtype": tf.int32})]
        # }
        ]
# hyperparameters
dimensions = [
    # architecture search
    skopt.space.Integer(0, 1, name='force_recursion'),
    skopt.space.Integer(0, 1, name='force_skip'),
    skopt.space.Integer(1, 3, name='initial_boxes'),
    skopt.space.Integer(1, 2, name='min_mutations'),
    skopt.space.Integer(1, 2, name='max_mutations'),
    skopt.space.Integer(1, 2, name='min_layers'),
    skopt.space.Integer(1, 4, name='max_layers'),
    skopt.space.Integer(1, 2, name='min_nodes'),
    skopt.space.Integer(1, 4, name='max_nodes'),
    skopt.space.Real(0.01, 0.1, name='p_insert'),
    # mutation distribution:
    skopt.space.Real(0, 1, name="p_recurse"),
    skopt.space.Real(0, 1, name="p_decimate"),
    skopt.space.Real(0, 1, name="p_connect"),
    skopt.space.Real(0, 1, name="p_split_edges"),
    skopt.space.Real(0, 1, name="p_insert_motif"),
    skopt.space.Real(0, 1, name="p_add_edge"),
    skopt.space.Real(0, 1, name="p_delete_edge"),
    skopt.space.Real(0, 1, name="p_delete_node"),
    skopt.space.Real(0, 1, name="p_split_edge"),
    skopt.space.Real(0, 1, name="p_do_nothing"),
    # node_type distribution:
    skopt.space.Real(0, 1, name="p_sepconv"),
    skopt.space.Real(0, 1, name="p_transformer"),
    skopt.space.Real(0, 1, name="p_k_conv1"),
    skopt.space.Real(0, 1, name="p_k_conv2"),
    skopt.space.Real(0, 1, name="p_k_conv3"),
    skopt.space.Real(0, 1, name="p_deep"),
    skopt.space.Real(0, 1, name="p_wide_deep"),
    # activation distribution:
    skopt.space.Real(0, 1, name="p_tanh"),
    skopt.space.Real(0, 1, name="p_linear"),
    skopt.space.Real(0, 1, name="p_relu"),
    skopt.space.Real(0, 1, name="p_selu"),
    skopt.space.Real(0, 1, name="p_elu"),
    skopt.space.Real(0, 1, name="p_sigmoid"),
    skopt.space.Real(0, 1, name="p_hard_sigmoid"),
    skopt.space.Real(0, 1, name="p_exponential"),
    skopt.space.Real(0, 1, name="p_softmax"),
    skopt.space.Real(0, 1, name="p_softplus"),
    skopt.space.Real(0, 1, name="p_softsign"),
    skopt.space.Real(0, 1, name="p_gaussian"),
    skopt.space.Real(0, 1, name="p_sin"),
    skopt.space.Real(0, 1, name="p_cos"),
    skopt.space.Real(0, 1, name="p_swish"),
    # layers:
    skopt.space.Integer(1, 4, name='min_filters'),
    skopt.space.Integer(1, 8, name='max_filters'),
    skopt.space.Integer(1, 8, name='min_units'),
    skopt.space.Integer(1, 128, name='max_units'),
    skopt.space.Integer(1, 2, name='attn_heads'),
    skopt.space.Integer(1, 2, name='min_k_layers'),
    skopt.space.Integer(1, 4, name='max_k_layers'),
    skopt.space.Real(0, 1, name='p_wide_deep'),
    skopt.space.Real(0.96, 1, name='p_force_residual'),
    skopt.space.Categorical(['sepconv1d', "k_conv1", "k_conv2", "k_conv3", "deep", "wide_deep", "transformer"],
                            name='last_layer'),
    skopt.space.Real(0, 100, name='output_gain'),
    skopt.space.Real(0.001, 0.1, name='stddev'),
    # optimizer:
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 10000000, name='decay')]


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(**kwargs):
    global run_step, best, best_trial, best_args, trial_number, writer, ema, agent, optimizer, hp
    start_time = time.perf_counter()
    # we get the agent
    hp = AttrDict(kwargs)
    trial_time = str(datetime.now()).replace(" ", "_")
    agent = Agent(hp, tasks, name=trial_time)
    # we get the optimizer and ema
    lr = tf.cast(hp.lr, tf.float32)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    ema.apply(agent.weights)
    lr = tf.keras.experimental.CosineDecayRestarts(lr, hp.decay)
    optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    writer = tf.summary.create_file_writer(log_dir)

    trainers = get_trainers(tasks)
    results = train(agent, trainers)

    mean = tf.math.reduce_mean(results)
    stddev = tf.math.reduce_std(results)
    objective = mean + stddev  # reward skill and consistency
    tf.summary.scalar('objective', objective)
    tf.print('mean:', mean, 'stddev:', stddev, 'objective:', objective)

    if tf.math.is_nan(objective):
        objective = 142.
    elif tf.math.less(objective, best):
        plot_path = os.path.join('.', 'runs', trial_name + '.png')
        K.utils.plot_model(agent, plot_path, show_shapes=True)
        agent.summary()
        averages = [ema.average(weight).numpy() for weight in agent.weights]
        agent.set_weights(averages)
        agent.save(os.path.join(log_dir, trial_name + ".h5"))
        best_trial = trial_name
        best_args = hp
        best = objective
        # make gifs
        run_agent_on_env(agent, "MolEnv-v0", n_episodes)

    print('best_trial', best_trial)
    print('best_args', best_args)
    print('best', best)
    del agent, writer
    trial_number += 1
    if ACQUISITION_FUNCTION is 'EIps':
        elapsed = time.perf_counter() - start_time
        print(f'trial {trial_number} done in {elapsed}S')
        return objective, elapsed
    else:
        return objective


class PlotCallback(object):
    def __init__(self, path):
        self.path = path

    def __call__(self, results):
        print(results)
        if trial_number > 10:
            _ = plot_evaluations(results)
            plt.savefig(os.path.join(self.path, "evaluations.png"))
            _ = plot_objective(results)
            plt.savefig(os.path.join(self.path, "objective.png"))


def experiment():
    global log_dir, proteins
    log_dir = os.path.join('.', 'runs', str(datetime.datetime.now()).replace(" ", "_"))

    if TENSORBOARD:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        webbrowser.get(using='google-chrome').open(tb.launch()+'#scalars', new=2)

    checkpoint_path = os.path.join(log_dir, 'checkpoint.pkl')
    checkpointer = skopt.callbacks.CheckpointSaver(checkpoint_path, compress=9)
    plotter = PlotCallback('.')
    try:
        res = skopt.load(checkpoint_path)
        x0 = res.x_iters
        y0 = res.func_vals
        results = skopt.gp_minimize(trial, dimensions, x0=x0, y0=y0,
                                    verbose=True, acq_func=ACQUISITION_FUNCTION,
                                    callback=[checkpointer, plotter])
    except Exception as e:
        print(e)

    n_protein_records, proteins = read_shards("cif")

    results = skopt.gp_minimize(trial, dimensions,
                                verbose=True, acq_func=ACQUISITION_FUNCTION,
                                callback=[checkpointer, plotter])
    print(results)


if __name__ == '__main__':
    experiment()
