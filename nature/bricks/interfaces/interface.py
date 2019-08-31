import tensorflow as tf

from nature import RaggedSensor, RaggedActuator, Brick
from nature import Resizer, Norm, Classifier
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


def Coder(agent, in_spec):
    normalizer = norm = Norm()

    if in_spec.rank is 3:
        h, w = get_hw(in_spec.shape)
        hw = [h, w]

        def resize_then_norm(x):
            x = tf.image.resize(x, hw)
            return norm(x)
        normalizer = resize_then_norm

    coordinator = L.Lambda(concat_coords)
    sensor, actuator = get_sensor_and_actuator(agent, in_spec)

    def call(self, x):
        normie = normalizer(x)
        normie_w_coords = coordinator(normie)
        code = sensor(normie_w_coords)
        reconstruction = actuator(code)
        self.add_loss(K.losses.MSLE(normie, reconstruction))
        return code
    return Brick(
        in_spec, norm, normalizer, coordinator, sensor, actuator, call, agent)


def Actuator(agent, spec):
    if spec.rank is 3 and spec.shape[1] is None:
        return RaggedActuator(agent, spec)
    elif spec.rank is 4:
        return ImageActuator(agent)
    elif spec.format in ['discrete', 'onehot']:
        return Classifier(agent, spec)
    else:
        return Resizer(spec.shape)
