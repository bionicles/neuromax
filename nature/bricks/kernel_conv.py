# conv-kernel.py
# why?: build a resnet with kernel and attention set convolutions

import tensorflow as tf
import random

from .mlp import get_mlp

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras


class KConvSet(L.Layer):
    def __init__(self, agent, d_in, d_out, set_size):
        self.kernel = get_mlp(agent, d_in, d_out, set_size)
        if set_size is 1:
            self.call = self.call_for_one
        elif set_size is 2:
            self.call = self.call_for_two
        elif set_size is 3:
            self.call = self.call_for_three
        super(KConvSet, self).__init__(name=f"KConvSet{set_size}-{random.randint(0, 9001)}")

    def call_for_one(self, atoms):
        return tf.map_fn(lambda a1: self.kernel(a1), atoms)

    def call_for_two(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), atoms), axis=0), atoms)

    def call_for_three(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: tf.reduce_sum(tf.map_fn(lambda a3: self.kernel([a1, a2, a3]), atoms), axis=0), atoms), axis=0), atoms)
