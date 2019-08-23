import tensorflow as tf

from nature import use_ragged_sensor, use_ragged_actuator
from nature import use_image_sensor, use_image_actuator
from nature import use_resizer, use_norm
from tools import concat_coords

K = tf.keras
L = K.layers


def get_sensor_and_actuator(agent, in_spec):
    if 4 in [in_spec.rank]:
        sensor = use_image_sensor(agent)
        actuator = use_image_actuator(agent)
    elif in_spec.rank is 3 and in_spec.shape[1] is None:
        sensor = use_ragged_sensor(agent, in_spec)
        actuator = use_ragged_actuator(agent, in_spec)
    else:
        sensor = use_resizer(agent.code_spec.shape)
        actuator = use_resizer(in_spec.shape)
    return sensor, actuator


def use_coder(agent, in_spec):
    norm = use_norm()
    coordinator = L.Lambda(concat_coords)
    sensor, actuator = get_sensor_and_actuator(agent, in_spec)

    def call(x):
        normie = norm(x)
        normie_w_coords = coordinator(normie)
        code = sensor(normie_w_coords)
        reconstruction = actuator(code)
        return normie, code, reconstruction
    return call

def use_actuator(agent, spec):
    if spec.rank is 3 and spec.shape[1] is None:
        return use_ragged_actuator(agent, spec)
    if spec.rank is 4:
        return use_image_actuator(agent)
    else:
        return use_resizer(spec.shape)
