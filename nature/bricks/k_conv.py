# conv-kernel.py
# why?: build a resnet with kernel and attention set convolutions

from nanoid import generate
import tensorflow as tf

from nature.bricks.kernel import get_kernel

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras


class KConvSet1D(L.Layer):
    """Convolve a learned kernel over sets of elements from a 1D tensor"""

    def __init__(self, agent, brick_id, in_spec, out_spec, set_size):
        self.brick_id = brick_id
        self.agent = agent
        d_in = in_spec.shape[-1]
        d_out = out_spec.shape[-1]
        if set_size is 1:
            self.call = self.call_for_one
        elif set_size is 2:
            self.call = self.call_for_two
        elif set_size is 3:
            self.call = self.call_for_three
        elif set_size is "all_for_one":  # ragged actuators
            self.call = self.call_all_for_one
            self.flatten = L.Flatten()
            d_in = 1
        elif set_size is "one_for_all":  # ragged sensors
            self.reshape = L.Reshape(out_spec.shape)
            self.call = self.call_one_for_all
            self.flatten = L.Flatten()
            d_out = out_spec.size
        print("kconvset1D init d_in", d_in, "d_out", d_out)
        self.kernel = get_kernel(agent, brick_id, d_in, d_out, set_size)
        self.layer_id = f"{brick_id}_KConvSet_{set_size}-{generate()}"
        super(KConvSet1D, self).__init__(name=self.layer_id)

    # TODO: find a nice recursive approach to N-ary set convolutions
    def call_for_one(self, atoms):
        return tf.map_fn(lambda a1: self.kernel(a1), atoms)

    def call_for_two(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), atoms), axis=0), atoms)

    def call_for_three(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: tf.reduce_sum(tf.map_fn(lambda a3: self.kernel([a1, a2, a3]), atoms), axis=0), atoms), axis=0), atoms)

    def call_one_for_all(self, inputs):  # input unknown needs ragged sensor
        """Each input element innervates all output elements"""
        input, output_placeholder = inputs
        output_placeholder = self.flatten(output_placeholder)
        print("call_one_for_all input", input,
              "output_placehodl", output_placeholder)
        output = tf.foldl(
            lambda output_placeholder, input_element:
                output_placeholder + self.kernel(
                    [input_element, output_placeholder]
                    ), input)
        return self.reshape(output)

    def call_all_for_one(self, inputs):  # output unknown needs ragged actuator
        """Each output element recieves all input elements"""
        normalized_output_coords, code = inputs
        flat_code = self.flatten(code)
        print("call_one_for_all", normalized_output_coords, flat_code)
        return tf.map_fn(lambda normalized_output_coord:
                         self.kernel([
                             normalized_output_coord, flat_code
                             ]), normalized_output_coords)

    # TOO COMPLICATED:
    # def call_autoregressive(self, code, coords):
    #     return tf.foldl(lambda done, coord: tf.concat(done,
    #         tf.reduce_mean(
    #             tf.map_fn(lambda done_item:
    #                 self.kernel([coord, done_item code]), coords), axis=0)), coords)
