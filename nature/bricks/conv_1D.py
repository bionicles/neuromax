import tensorflow as tf

MIN_FILTERS, MAX_FILTERS = 4, 16
ACTIVATIONS = ["tanh"]

L = tf.keras.layers


def get_conv_1D(agent, brick_id, d_out=None, activation=None):
    if d_out is None:
        d_out = agent.pull_numbers(f"{id}_filters", MIN_FILTERS, MAX_FILTERS)
    if activation is None:
        activation = agent.pull_choices(f"{id}_activation", ACTIVATIONS)
    brick = L.SeparableConv1D(d_out, 1, activation=activation)
    return brick
