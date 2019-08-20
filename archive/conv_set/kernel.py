import tensorflow as tf

from tools import make_uuid, log
from nature import use_dense

K = tf.keras
L = K.layers

MODEL_OPTIONS = ["deep", "wide_deep"]
MIN_LAYERS, MAX_LAYERS = 1, 2
SET_OPTIONS = [-1, 1]


def use_kernel(agent, id, d_in, d_out, set_size, input_shape=None, d_in2=None):
    """get a deep or wide/deep dense set kernel"""
    log("get_kernel", id, d_in, d_out, set_size)
    log("agent code spec", agent.code_spec)
    name = make_uuid([id, "kernel"])
    n_layers = agent.pull_numbers(f"{name}-n_layers", MIN_LAYERS, MAX_LAYERS)
    model_type = agent.pull_choices(f"{name}-model_type", MODEL_OPTIONS)
    if set_size is None:
        set_size = agent.pull_choices(f"{name}-set_size", SET_OPTIONS)
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
    elif set_size is "all_for_one":
        atom1 = K.Input((d_in + 1, 1))
        inputs = [atom1]
        concat = atom1
    elif set_size is "one_for_all":
        code = K.Input((d_in2,))
        inputs = [atom1, code]
        concat = L.Concatenate(-1)([atom1, code])
    output = use_dense(agent, f"{name}_0")(concat)
    for i in range(n_layers - 1):
        output = use_dense(agent, f"{name}_{i}")(output)
    if "wide" in model_type:
        stuff_to_concat = inputs + [output]
        output = L.Concatenate(-1)(stuff_to_concat)
    output = use_dense(agent, f"{name}_{n_layers}", units=d_out)(output)
    name = f"{name}_{n_layers}_{model_type}"
    return K.Model(inputs, output, name=name)
