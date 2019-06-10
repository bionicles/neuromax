# agent.py why?: solve environments with rl
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from resnet import make_resnet
from tensorflow.keras import Model
import tensorflow as tf
from queue import Queue
import numpy as np
import threading
import os


class ActorCritic(Model):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.policy_resnet = make_resnet(config)
        self.critic_resnet = make_resnet(config)
        self.policy_logits = Dense(config.ACTION_DIMENSION, activation='relu')
        self.values = Dense(1, activation='relu')

    def call(self, inputs):
        # Forward pass
        before_logits = self.policy_resnet(inputs)
        policy = self.policy_logits(before_logits)
        before_critic = self.critic_resnet(inputs)
        critic = self.values(before_critic)
        return policy, critic


class MasterAgent():
    def __init__(self, config, env):
        self.env = env
        self.config = config
        save_dir = self.config.SAVE_DIR
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.opt = tf.train.AdamOptimizer(self.config.LR, use_locking=True)
        self.global_model = ActorCritic(self.config)  # global network

    def train(self):
        res_queue = Queue()
        workers = [Worker(self.config, self.global_model,
                          self.opt, res_queue, self.env,
                          i) for i in range(self.config.NUM_WORKERS)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.run()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(
            os.path.join(self.save_dir, '{} MA.png'.format(self.game_name)))
        plt.show()


class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, config, global_model, opt, result_queue, env, idx):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.global_model = global_model
        self.config = config
        self.opt = opt
        self.local_model = ActorCritic()
        self.worker_idx = idx
        self.env = env
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.config.NUM_EPISODES:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            time_count = 0
            done = False
            while not done:
                action, _ = self.local_model(current_state)
                new_state, reward, done, _ = self.env.step(action)
            if done:
                reward = -1
            ep_reward += reward
            mem.store(current_state, action, reward)

            if time_count == self.config.UPDATE_FREQ or done:
                # Calculate gradient wrt to local model. We track the
                # variables with tf.GradientTape
                with tf.GradientTape() as tape:
                    total_loss = self.compute_loss(done, new_state, mem)
                self.ep_loss += total_loss
                # Calculate local gradients
                grads = tape.gradient(
                    total_loss, self.local_model.trainable_weights)
                # Push local gradients to global model
                self.opt.apply_gradients(
                    zip(grads, self.global_model.trainable_weights))
                # Update local model with new weights
                self.local_model.set_weights(self.global_model.get_weights())

                mem.clear()
                time_count = 0

            if done:
                Worker.global_moving_average_reward = \
                    record(
                        Worker.global_episode, ep_reward, self.worker_idx,
                        Worker.global_moving_average_reward, self.result_queue,
                        self.ep_loss, ep_steps)
                # We use a lock to save and print to prevent data races.
                if ep_reward > Worker.best_score:
                    with Worker.save_lock:
                        print(
                            "Saving best model to {}, ",
                            "episode score: {}".format(
                                self.config.SAVE_DIR, ep_reward))

                    self.global_model.save_weights(
                        os.path.join(self.config.save_dir,
                                     'model.h5')
                    )
                    Worker.best_score = ep_reward
                Worker.global_episode += 1
            ep_steps += 1

            time_count += 1
            current_state = new_state
            total_step += 1
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            _, reward_sum = self.local_model(new_state)

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        policy, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(
            np.array(discounted_rewards)[:, None],
            dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(
            memory.actions, self.config.ACTION_DIMENSION, dtype=tf.float32)

        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)

        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=actions_one_hot,
            logits=policy)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
