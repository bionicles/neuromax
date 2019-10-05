# conv-kernel.py
# why?: build a resnet with kernel and attention set convolutions
import tensorflow as tf

from .kernel import get_kernel

from tools import concat_1D_coords, log, make_uuid

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras

# todo ... fix one_for_all and all_for_one inside graph_model
SET_OPTIONS = [-1, 1, 2, 3]


class KConvSet1D(L.Layer):
    """Convolve a learned kernel over sets of elements from a 1D tensor"""

    def __init__(self, agent, brick_id, in_spec, out_spec, set_size):
        self.out_spec = out_spec
        self.in_spec = in_spec
        d_out = out_spec if isinstance(out_spec, int) else out_spec.shape[-1]
        d_in = in_spec if isinstance(in_spec, int) else in_spec.shape[-1]
        d_in2 = None
        if set_size is None:
            set_size = agent.pull_choices(f"{brick_id}_KConvSet_set_size",
                                          SET_OPTIONS)
        self.brick_id = brick_id
        self.agent = agent
        if set_size is 1:
            self.call = self.call_for_one
        elif set_size is 2:
            self.call = self.call_for_two
        elif set_size is 3:
            self.call = self.call_for_three
        elif set_size is "one_for_all":  # ragged sensors
            # d_in is size of 1 element of shape
            # d_in2 is code_spec.size
            d_out = out_spec.size
            set_size = 1
            self.reshape = L.Reshape(out_spec.shape)
            self.call = self.call_one_for_all
            self.flatten = L.Flatten()
        elif set_size is "all_for_one":  # ragged actuators
            d_in = in_spec.size
            # d_in2 is 1 (placeholder range)
            # d_out is size of 1 element of the output shape
            self.call = self.call_all_for_one
        self.kernel = get_kernel(agent, brick_id, d_in, d_out, set_size,
                                 d_in2=d_in2)
        self.id = make_uuid([id, "KConvSet1D", set_size])
        super(KConvSet1D, self).__init__(name=self.id)

    def call_one_for_all(self, input):  # input unknown = ragged sensor
        """Each input element innervates all output elements"""
        # output = tf.foldl(lambda a, item: a + self.kernel(item), input)
        output = tf.map_fn(lambda item: self.kernel(item), input)
        log("call_one_for_all", output.shape, color="blue")
        output = tf.math.reduce_sum(output, axis=1)
        log("call_one_for_all", output.shape, color="blue")
        return output

    def call_all_for_one(self, inputs):  # output unknown = ragged actuator
        """All input elements innervate each output element"""
        # log("")
        # log("we map all inputs onto one output", color="green")
        # log(f"[{self.in_spec.shape}] in_spec", color="white")
        code, placeholder = inputs
        placeholder_with_coords = concat_1D_coords(placeholder)
        placeholder_with_coords = tf.squeeze(placeholder_with_coords, 0)
        coords = tf.slice(placeholder_with_coords, [0, 1], [-1, 1])
        code = tf.squeeze(code, 0)
        log(list(code.shape), "code", color="white")
        log(list(coords.shape), "coords", color="white")
        output = tf.map_fn(
            lambda coord: self.kernel(
                tf.concat([code, coord], 0)
                ), tf.expand_dims(coords, 1))
        # log(f"[{self.out_spec.shape}] out_spec", color="blue")
        output = tf.reduce_sum(output, 1)
        # log(list(output.shape), "output", color="yellow")
        # log("")
        return output

    # TODO: find a nice recursive approach to N-ary set convolutions
    def call_for_one(self, atoms):
        return tf.map_fn(lambda a1: self.kernel(a1), atoms)

    def call_for_two(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), atoms), axis=0), atoms)

    def call_for_three(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: tf.reduce_sum(tf.map_fn(lambda a3: self.kernel([a1, a2, a3]), atoms), axis=0), atoms), axis=0), atoms)

    # TOO COMPLICATED:
    # def call_autoregressive(self, code, coords):
    #     return tf.foldl(lambda done, coord: tf.concat(done,
    #         tf.reduce_mean(
    #             tf.map_fn(lambda done_item:
    #                 self.kernel([coord, done_item code]), coords), axis=0)), coords)
