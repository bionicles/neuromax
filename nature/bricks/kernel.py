from blessings import Terminal
import tensorflow as tf

from nature.bricks.dense import get_dense_out
from tools.get_unique_id import get_unique_id
from tools.log import log



K = tf.keras
L = K.layers
T = Terminal()

SET_OPTIONS = [-1, 1, 2, 3, "all_for_one", "one_for_all"]
MODEL_OPTIONS = ["deep", "wide_deep"]
MIN_LAYERS, MAX_LAYERS = 1, 4


def get_kernel(agent, brick_id, d_in, d_out, set_size,
               name=None, input_shape=None):
    """get a deep or wide/deep dense set kernel"""
    log("get_kernel", brick_id, d_in, d_out, set_size)
    assert set_size in SET_OPTIONS, f"{set_size} not in {SET_OPTIONS}"
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
    elif set_size in ["all_for_one", "one_for_all"]:
        print(T.green("get_kernel set_size"), set_size)
        d_in2 = agent.code_spec.size
        code = K.Input((d_in2,))
        inputs = [atom1, code]
        concat = L.Concatenate(-1)([atom1, code])
    print(T.green("get_kernel concat"), concat)
    output = get_dense_out(agent, f"{name}_0", concat)
    for i in range(n_layers - 1):
        output = get_dense_out(agent, f"{name}_{i}", output)
    if model_type == "wide_deep":
        stuff_to_concat = inputs + [output]
        output = L.Concatenate(-1)(stuff_to_concat)
    print("last layer...", output)
    output = get_dense_out(agent, f"{name}_dense_{n_layers}", output, units=d_out)
    name = f"{name}_{n_layers}_{model_type}"
    return K.Model(inputs, output, name=name)
