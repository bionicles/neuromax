from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import BatchNormalization

BATCH = False


def use_norm():
    if BATCH:
        return BatchNormalization()
    return InstanceNormalization()
