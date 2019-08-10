import tensorflow as tf

from nature.bricks import get_layer
from tools.get_unique_id import get_unique_id
from tools.log import log

K = tf.keras
L = K.layers


MIN_LAYERS, MAX_LAYERS = 1, 4
MODEL_OPTIONS = ["deep", "wide_deep"]


def get_kernel(agent, brick_id, d_in, d_out, set_size,
               name=None, input_shape=None):
    """build a deep or wide and deep dense mlp"""
    log("get_kernel", brick_id, d_in, d_out, set_size)
    assert set_size in [-1, 1, 2, 3]
    name = get_unique_id(f"{brick_id}_mlp") if name is None else name
    n_layers = agent.pull_numbers(f"{name}-n_layers", MIN_LAYERS, MAX_LAYERS)
    model_type = agent.pull_choices(f"{name}-model_type", MODEL_OPTIONS)
    atom1 = K.Input((d_in, ))
    if set_size is -1:
        if input_shape is None:
            input_shape = (None, d_in)
        inputs = [K.Input(input_shape)]
        concat = inputs[0]
    elif set_size is 1:
        inputs = [atom1]
        concat = atom1
    elif set_size is 2:
        atom2 = K.Input((d_in, ))
        inputs = [atom1, atom2]
        d12 = L.Subtract()([atom1, atom2])
        concat = L.Concatenate(-1)([d12, atom1, atom2])
    elif set_size is 3:
        atom2 = K.Input((d_in, ))
        atom3 = K.Input((d_in, ))
        inputs = [atom1, atom2, atom3]
        d12 = L.Subtract()([atom1, atom2])
        d13 = L.Subtract()([atom1, atom3])
        concat = L.Concatenate(-1)([d12, d13, atom1, atom2, atom3])
    lkey = f"{name}_0"
    output = get_layer(agent, lkey)(concat)
    for i in range(len(n_layers) - 1):
        lkey = f"{name}_{i}"
        output = get_layer(agent, lkey)(output)
    if model_type == "wide_deep":
        stuff_to_concat = inputs + [output]
        output = L.Concatenate(-1)(stuff_to_concat)
    lkey = f"{name}_{n_layers}"
    output = get_layer(agent, lkey)(output)
    name = f"{name}_{n_layers}_{model_type}"
    return K.Model(inputs, output, name=name)
