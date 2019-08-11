import tensorflow_probability as tfp
from blessings import Terminal
import tensorflow as tf

from nature.bricks.activations import clean_activation

# tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
B, L = K.backend, K.layers
T = Terminal()

FN_OPTIONS = ["tanh", "linear", "swish", "lisht", "sigmoid"]
MIN_STDDEV, MAX_STDDEV = 1e-4, 0.1
MIN_UNITS, MAX_UNITS = 32, 512
P_DROP = 0.5


class NoiseDrop(L.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoiseDrop, self).__init__(*args, **kwargs)

    @tf.function
    def add_noise(self):
        return (
            self.kernel + tf.random.truncated_normal(
                tf.shape(self.kernel), stddev=self.stddev),
            self.bias + tf.random.truncated_normal(
                tf.shape(self.bias), stddev=self.stddev))

    def call(self, x):
        kernel, bias = self.add_noise()
        return self.activation(
                    tf.nn.bias_add(
                        B.dot(x, tf.nn.dropout(kernel, P_DROP)), bias))


LAYER_OPTIONS = [NoiseDrop, L.Dense, tfpl.DenseFlipout,
                 tfpl.DenseReparameterization]


def get_dense_out(agent, brick_id, input, layer=None, units=None, fn=None,
                  fn_options=None):
    print(T.blue("get_dense_out"), brick_id, input, layer, units)
    if layer is None:
        layer = agent.pull_choices(f"{brick_id}-layer_type", LAYER_OPTIONS)
    if units is None:
        units = agent.pull_numbers(f"{brick_id}-units", MIN_UNITS, MAX_UNITS)
    if fn is None:
        if fn_options is None:
            fn = agent.pull_choices(f"{brick_id}-fn", FN_OPTIONS)
        else:
            fn = agent.pull_choices(f"{brick_id}-fn", fn_options)
    fn = clean_activation(fn)
    if layer is NoiseDrop:
        stddev = agent.pull_numbers(f"{brick_id}-stddev",
                                    MIN_STDDEV, MAX_STDDEV)
        return NoiseDrop(units, activation=fn, stddev=stddev)(input)
    else:
        return layer(units, activation=fn)(input)
    raise Exception(f"get_layer failed on {brick_id} {layer} {units} {fn}")