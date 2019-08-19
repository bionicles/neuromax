import tensorflow as tf

from .dense import get_dense

K = tf.keras
L = K.layers

MIN_POWER, MAX_POWER = 2, 3


# functional style
def get_swag_out(agent, brick_id, units, input, power=None):
    if power is None:
        power = agent.pull_numbers(f"{brick_id}_power", MIN_POWER, MAX_POWER)
    if "swag" not in brick_id:
        brick_id = f"{brick_id}_swag_{power}"
    outputs = []
    for _ in range(power):
        layer = get_dense(agent, brick_id)
        if len(outputs) > 0:
            output = layer(outputs[-1])
            output = L.Multiply()([outputs[-1], output])
        else:
            output = layer(input)
        outputs.append(output)
    flatten = L.Flatten()
    outputs = [flatten(o) for o in outputs]
    output = L.Concatenate()(outputs)
    output = get_dense(agent, brick_id, units=units)(output)
    return output


# object-oriented style
def get_swag(agent, brick_id, units, input_shape, power=None):
    brick_id = f"{brick_id}_swag{'' if power is None else '_' + str(power)}"
    input = K.Input(input_shape[1:])
    output = get_swag_out(agent, brick_id, units, input, power=power)
    return K.Model(input, output, name=brick_id)
