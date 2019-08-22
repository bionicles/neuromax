from tensorflow.keras import Input


def use_input(shape_or_tensor, batch_size=1):
    if hasattr(shape_or_tensor, "shape"):
        shape = shape_or_tensor.shape
    else:
        shape = shape_or_tensor
    return Input(shape[1:], batch_size=batch_size)
