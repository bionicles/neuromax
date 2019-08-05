import tensorflow as tf

from . import get_layer
from helpers import get_name

K = tf.keras
L = K.layers


MIN_LAYERS, MAX_LAYERS = 2, 2


def get_mlp(agent, d_in, d_out, set_size, name=None):
    """build a deep or wide and deep dense mlp"""
    assert set_size in [-1, 1, 2, 3]
    name = get_name("mlp") if name is None else name
    n_layers = agent.pull_numbers(f"{name}-n_layers", MIN_LAYERS, MAX_LAYERS)
    model_type = agent.pull_choices(f"{name}-model_type", ["deep", "wide_deep"])
    atom1 = K.Input((d_in, ))
    if set_size is -1:
        atom1 = K.Input((None, d_in))
        inputs = [atom1]
        concat = atom1
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
    lkey = f"{name}-0"
    output = get_layer(lkey, agent)(concat)
    for i in range(len(n_layers) - 1):
        lkey = f"{name}-{i}"
        output = get_layer(lkey, agent)(output)
    if model_type == "wide_deep":
        stuff_to_concat = inputs + [output]
        output = L.Concatenate(-1)(stuff_to_concat)
    lkey = f"{name}-{n_layers}"
    output = get_layer(lkey, agent)(output)
    name = f"{name}_{n_layers}_{model_type}"
    return K.Model(inputs, output, name=name)
