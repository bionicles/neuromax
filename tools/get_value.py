def get_value(maybe_tensor):
    if isinstance(maybe_tensor, int) or isinstance(maybe_tensor, float):
        return maybe_tensor
    else:
        return maybe_tensor.numpy().item(0)
