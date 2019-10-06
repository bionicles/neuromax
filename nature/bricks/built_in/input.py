import tensorflow as tf
from tools import make_id


def Input(spec, batch_size=1, drop_batch_dim=False):
    shape, format = spec["shape"], spec["format"]
    # if tf.is_tensor(tensor_or_shape):
    #     shape = tensor_or_shape.shape
    # else:
    #     shape = tensor_or_shape
    # if drop_batch_dim:
    #     shape = shape[1:]
    shape_string = 'x'.join([str(n) for n in list(shape)])
    name = make_id(f"{batch_size}x{shape_string}_{format}")
    return tf.keras.Input(
        shape, batch_size=batch_size, dtype=tf.float32, name=name)
