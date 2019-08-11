# -*- coding: utf-8 -*-
# pylint: disable=W0221
"""
Differentiable Neural Computer model definition.
Reference:
    http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html
Conventions:
    B - batch size
    N - number of slots in memory
    R - number of read heads
    W - size of each memory slot i.e word size
"""

from collections import namedtuple, OrderedDict
from tensorflow.python.util import nest
import tensorflow_probability as tfp
import tensorflow as tf

from nature.bricks.dnc.memory import Memory, EPSILON

K = tf.keras
L = K.layers
tfpl = tfp.layers

CONTROLLER_UNITS = 256
MEMORY_SIZE = 256
WORD_SIZE = 64
NUM_READ_HEADS = 4
TFP_LAYER = tfpl.DenseFlipout
DTYPE = tf.float32


class DNC_Cell(L.Layer):
    """DNC cell module that connects together the controller and memory.
    Performs a write and read operation against memory given
    1) the previous state
    and
    2) an interface vector defining how to interact with the memory at the
    current time step.
    Args:
        output_size (int): size of final output dimension for the whole DNC cell at each time step
        controller_units (int): size of hidden state in controller
        memory_size (int): number of slots in external memory
        word_size (int): the width of each memory slot
        num_read_heads  (int): number of memory read heads
    """

    state = namedtuple("dnc_state", [
        "memory_state",
        "controller_state",
        "read_vectors"])

    interface = namedtuple("interface", [
        "read_keys",
        "read_strengths",
        "write_key",
        "write_strength",
        "erase_vector",
        "write_vector",
        "free_gates",
        "allocation_gate",
        "write_gate",
        "read_modes"])

    def __init__(self, agent, brick_id, output_size, controller_units=CONTROLLER_UNITS, memory_size=MEMORY_SIZE,
                 word_size=WORD_SIZE, num_read_heads=NUM_READ_HEADS, tfp_layer=TFP_LAYER, **kwargs):
        super().__init__(name='DNC_Cell', **kwargs)
        self.agent = agent
        self.brick_id = brick_id
        self.pull_choices = agent.pull_choices
        self.pull_numbers = agent.pull_numbers
        self._output_size = output_size
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads
        self.word_size = word_size
        self._interface_vector_size = self.num_read_heads * self.word_size
        self._interface_vector_size += 3 * self.word_size
        self._interface_vector_size += 5 * self.num_read_heads + 3
        self._clip = 20.0
        self._controller = L.LSTMCell(units=controller_units)
        if tfp_layer:
            self._controller_to_interface_dense = tfp_layer(
                self.interface_vector_size, name='controller_to_interface')
            self._final_output_dense = tfp_layer(self._output_size)
        else:
            self._controller_to_interface_dense = L.Dense(
                self._interface_vector_size, name='controller_to_interface')
            self._memory = Memory(memory_size, word_size, num_read_heads)
            self._final_output_dense = L.Dense(self._output_size)

    def _parse_interface_vector(self, interface_vector):
        r, w = self.num_read_heads, self.word_size
        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        fns = OrderedDict([
            ("read_keys", lambda v: tf.reshape(v, (-1, w, r))),
            ("read_strengths", lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, r))))),
            ("write_key", lambda v: tf.reshape(v, (-1, w, 1))),
            ("write_strength", lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, 1))))),
            ("erase_vector", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, w)))),
            ("write_vector", lambda v: tf.reshape(v, (-1, w))),
            ("free_gates", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, r)))),
            ("allocation_gate", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)))),
            ("write_gate", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)))),
            ("read_modes", lambda v: tf.nn.softmax(tf.reshape(v, (-1, 3, r)), axis=1))])
        indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
        zipped_items = zip(fns.keys(), fns.values(), indices)
        interface = {name: fn(interface_vector[:, i[0]:i[1]]) for name, fn, i in zipped_items}
        return DNC_Cell.interface(**interface)

    def _flatten_read_vectors(self, x):
        return tf.reshape(x, (-1, self.word_size * self.num_read_heads))

    def call(self, inputs, prev_dnc_state):
        prev_dnc_state = nest.pack_sequence_as(self.state_size_nested, prev_dnc_state)
        # inputs_to_controller
        read_vectors_flat = self._flatten_read_vectors(prev_dnc_state.read_vectors)
        input_augmented = tf.concat([inputs, read_vectors_flat], 1)
        controller_output, controller_state = self._controller(
            input_augmented,
            prev_dnc_state.controller_state,)
        controller_output = tf.clip_by_value(controller_output, -self._clip, self._clip)
        # parse_interface
        interface = self._controller_to_interface_dense(controller_output)
        interface = self._parse_interface_vector(interface)
        # update_memory
        read_vectors, memory_state = self._memory(interface, prev_dnc_state.memory_state)
        state = DNC_Cell.state(memory_state=memory_state,
                          controller_state=controller_state,
                          read_vectors=read_vectors)
        # join_outputs
        read_vectors_flat = self._flatten_read_vectors(read_vectors)
        final_output = tf.concat([controller_output, read_vectors_flat], 1)
        final_output = self._final_output_dense(final_output)
        final_output = tf.clip_by_value(final_output, -self._clip, self._clip)
        return final_output, nest.flatten(state)

    @property
    def state_size_nested(self):
        return DNC_Cell.state(
            memory_state=self._memory.state_size,
            controller_state=self._controller.state_size,
            read_vectors=tf.TensorShape([self.word_size, self.num_read_heads]))

    @property
    def state_size(self):
        return nest.flatten(self.state_size_nested)

    def get_initial_state(self, inputs=None, batch_size=1, dtype=DTYPE):
        del inputs
        initial_state_nested = DNC_Cell.state(
            memory_state=self._memory.get_initial_state(batch_size, dtype=dtype),
            controller_state=self._controller.get_initial_state(batch_size=batch_size, dtype=dtype),
            read_vectors=tf.fill([batch_size, self.word_size, self.num_read_heads], EPSILON))
        return nest.flatten(initial_state_nested)

    @property
    def output_size(self):
        return self._output_size

    def get_config(self):
        config = {
            'controller': {
                'class_name': self._controller.__class__.__name__,
                'config': self._controller.get_config()
            },
            'memory': {
                'read_heads_num': self.num_read_heads,
                'word_size': self.word_size,
                'words_num': self.memory_size,
            },
            'clip': self._clip,
            'output_size': self._output_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
