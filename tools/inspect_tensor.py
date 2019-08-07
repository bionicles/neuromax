import tensorflow as tf


def inspect_tensor(spec, tensor):
    tensor_shape = tf.shape(tensor)
    for dimension_number, dimension_value in enumerate(spec.shape):
        if dimension_value is not None:
            tensor_dimension = tensor_shape[dimension_number]
            if tensor_dimension != dimension_value:
                print(f"tensor failed inspection on dimension {dimension_number} {tensor_dimension} != {dimension_value}")
                raising = True
    if spec.format is "onehot":
        if tf.math.reduce_sum(tensor) > 1:
            print(f"spec calls for onehot but tensor sum > 1")
    if raising:
        raise Exception(f"{tensor} does not meet {spec}")
    return True
