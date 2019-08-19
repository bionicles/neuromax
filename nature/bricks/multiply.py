import tensorflow as tf

from .dense import get_dense

K = tf.keras
L = K.layers

ADD = True


def get_multiply_out(agent, brick_id, units, input):
    if "multiply" not in brick_id:
        brick_id = f"{brick_id}_multiply"
    o1 = get_dense(agent, brick_id, units=units)(input)
    o2 = get_dense(agent, brick_id, units=units)(input)
    out = L.Multiply()([o1, o2])
    if ADD:
        o3 = get_dense(agent, brick_id, units=units)(out)
        out = L.Add()([o3, out])
    return out


def get_multiply(agent, brick_id, units, in_shape):
    brick_id = f"{brick_id}_multiply"
    input = K.Input(in_shape[1:], batch_size=agent.batch_size)
    out = get_multiply_out(agent, brick_id, units, input)
    return K.Model(input, out, name=f"{brick_id}_multiply")
