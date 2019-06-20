# neuromax.py - why?: 1 simple file with functions over classes
import tensorflow.keras.layers as L
import tensorflow.keras as K
import tensorflow as tf
import bayes_opt
import random
import time
import csv
import os

# search
NUM_RANDOM_TRIALS, NUM_TRIALS = 4, 6
STOP_LOSS_MULTIPLIER = 1.04
NUM_EXPERIMENTS = 1000
NUM_EPISODES = 10
NUM_STEPS = 100

# hyperparameters
COMPLEXITY_PUNISHMENT = 1e-5  # 0 is off, higher is simpler
pbounds = {
    'GAIN': (1e-4, 0.1),
    'UNITS': (17, 2000),
    'LR': (1e-4, 1e-1),
    'EPSILON': (1e-4, 1),
    'LAYERS': (1, 10),
    'BLOCKS': (1, 50)
}


def train(GAIN, UNITS, LR, EPSILON, LAYERS, BLOCKS, L1, L2):
    start = time.time()
    TIME = str(start)
    run_path = os.path.join(ROOT, 'runs', TIME)
    if SAVE_MODEL:
        os.makedirs(run_path)
        save_path = os.path.join(run_path, 'model.h5')
    global_step = 0
    agent = make_resnet('agent', 16, 3, units=UNITS, blocks=BLOCKS,
                        layers=LAYERS, gain=GAIN, l1=L1, l2=L2)
    decayed_lr = tf.train.exponential_decay(LR, global_step, 10000, 0.96, staircase=True)
    adam = tf.train.AdamOptimizer(decayed_lr, epsilon=EPSILON)
    cumulative_improvement, episode = 0, 0
    load_pedagogy()
    for i in range(NUM_EPISODES):
        print('')
        print('BEGIN EPISODE', episode)
        done, step = False, 0
        screenshot = episode > WARMUP and episode % SCREENSHOT_EVERY == 0 and SAVE_MODEL
        initial, current, features = env.reset()
        initial_loss = loss(tf.zeros_like(positions), initial)
        stop_loss = initial_loss * STOP_LOSS_MULTIPLIER
        step += 1
        while not done:
            print('')
            print('experiment', experiment, 'model', TIME, 'episode', episode, 'step', step)
            print('BLOCKS', round(BLOCKS), 'LAYERS', round(LAYERS), 'UNITS', round(UNITS), 'LR', LR, 'L1', L1, 'L2', L2)
            with tf.GradientTape() as tape:
                atoms = tf.expand_dims(tf.concat([current, features], 1), 0)
                noise = tf.expand_dims(random_normal((num_atoms, 3)), 0)
                force_field = tf.squeeze(agent([atoms, noise]), 0)
                loss_value = loss(force_field, initial)
            gradients = tape.gradient(loss_value, agent.trainable_weights)
            adam.apply_gradients(zip(gradients, agent.trainable_weights))
            global_step += 1
            if screenshot:
                make_image()
            step += 1
            done_because_step = step > NUM_STEPS
            done_because_loss = loss_value > stop_loss
            done = done_because_step or done_because_loss
            if not done:
                current_stop_loss = loss_value * STOP_LOSS_MULTIPLIER
                stop_loss = current_stop_loss if current_stop_loss < stop_loss else stop_loss
        reason = 'STEP' if done_because_step else 'STOP LOSS'
        print('done because of', reason)
        percent_improvement = (initial_loss - loss_value) / 100
        cumulative_improvement += percent_improvement
        if screenshot:
            make_gif()
        episode += 1
        if SAVE_MODEL:
            tf.saved_model.save(agent, save_path)
    if COMPLEXITY_PUNISHMENT is not 0:
        cumulative_improvement /= agent.count_params() * COMPLEXITY_PUNISHMENT
    if TIME_PUNISHMENT is not 0:
        elapsed = time.time() - start
        cumulative_improvement /= elapsed * TIME_PUNISHMENT
    cumulative_improvement /= NUM_EPISODES
    return cumulative_improvement


def trial(GAIN, UNITS, LR, EPSILON, LAYERS, BLOCKS, L1, L2):
    tf.keras.backend.clear_session()
    try:
        return train(GAIN, UNITS, LR, EPSILON, LAYERS, BLOCKS, L1, L2)
    except Exception as e:
        print('EXPERIMENT FAIL!!!', e)
        return -100


def main():
    tf.enable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = bayes_opt.observer.JSONLogger(path='./runs/logs.json')
    bayes = bayes_opt.BayesianOptimization(f=partial(trial), pbounds=pbounds,
                                    verbose=2, random_state=1)
    bayes.subscribe(bayes_opt.event.Events.OPTMIZATION_STEP, logger)
    try:
        bayes_opt.util.load_logs(bayes, logs=['./runs/logs.json'])
    except Exception as e:
        print('failed to load bayesian optimization logs', e)
    for exp in range(NUM_EXPERIMENTS):
        bayes.maximize(init_points=NUM_RANDOM_TRIALS, n_iter=NUM_TRIALS)
        print("BEST MODEL:", bayes.max)
    NUM_RANDOM_TRIALS, NUM_TRIALS, NUM_EPISODES, SAVE_MODEL = 0, 1, 10000, True
    RANDOM_PROTEINS = True
    bayes.maximize(init_points=NUM_RANDOM_TRIALS, n_iter=NUM_TRIALS)
    print(bayes.res)

if __name__ == '__main__':
    main()
