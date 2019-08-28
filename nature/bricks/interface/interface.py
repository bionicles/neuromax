import tensorflow as tf

from nature import use_ragged_sensor, use_ragged_actuator
from nature import use_resizer, use_norm, add_classifier
from nature import use_image_sensor, use_image_actuator
from tools import concat_coords, log, get_hw

K = tf.keras
L = K.layers


def get_sensor_and_actuator(agent, in_spec):
    if in_spec.rank is 3:
        sensor = use_image_sensor(agent)
        actuator = use_image_actuator(agent)
    elif in_spec.rank is 2 and in_spec.shape[1] is None:
        sensor = use_ragged_sensor(agent, in_spec)
        actuator = use_ragged_actuator(agent, in_spec)
    else:
        sensor = use_resizer(agent.code_spec.shape)
        actuator = use_resizer(in_spec.shape)
    return sensor, actuator


def use_coder(agent, in_spec):
    log('use_coder', in_spec, color="blue")
    norm = use_norm()
    normalizer = norm

    if in_spec.rank is 3:
        h, w = get_hw(in_spec.shape)
        hw = [h, w]

        def resize_then_norm(x):
            x = tf.image.resize(x, hw)
            return norm(x)
        normalizer = resize_then_norm

    coordinator = L.Lambda(concat_coords)
    sensor, actuator = get_sensor_and_actuator(agent, in_spec)

    def call(x):
        normie = normalizer(x)
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
        call = use_resizer(spec.shape)
        if spec.format in ['discrete', 'onehot']:
            call = add_classifier(call, spec)
        return call
