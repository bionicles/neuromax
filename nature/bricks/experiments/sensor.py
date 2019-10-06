import tensorflow as tf

from nature import RaggedSensor, RaggedActuator, Resizer, Norm
from nature import ImageSensor, ImageActuator
from tools import concat_coords, get_hw

K = tf.keras
L = K.layers


def get_sensor_and_actuator(agent, in_spec):
    if in_spec.rank is 3:
        sensor = ImageSensor(agent)
        actuator = ImageActuator(agent)
    elif in_spec.rank is 2 and in_spec.shape[1] is None:
        sensor = RaggedSensor(agent, in_spec)
        actuator = RaggedActuator(agent, in_spec)
    else:
        sensor = Resizer(agent.code_spec.shape)
        actuator = Resizer(in_spec.shape)
    return sensor, actuator


class Sensor(L.Layer):
    def __init__(self, agent, in_spec):
        self.normalizer = norm = Norm()

        if in_spec.rank is 3:
            h, w = get_hw(in_spec.shape)
            hw = [h, w]

            def resize_then_norm(x):
                x = tf.image.resize(x, hw)
                return norm(x)
            self.normalizer = resize_then_norm

        self.coordinator = L.Lambda(concat_coords)
        self.sensor, self.actuator = get_sensor_and_actuator(agent, in_spec)

    def call(self, x):
        normie = self.normalizer(x)
        normie_w_coords = self.coordinator(normie)
        code = self.sensor(normie_w_coords)
        reconstruction = self.actuator(code)
        self.add_loss(K.losses.MSLE(normie, reconstruction))
        return code
