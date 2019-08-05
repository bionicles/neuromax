def map_enumerate(maybe_fn_or_fn_list, maybe_input_or_inputs, *args, **kwargs):
    """
    Apply a function or a list of functions to an input or list of inputs
    Args:
    maybe_fn_or_fn_list: a function or a list of functions
    maybe_input_or_inputs: an input or a list of inputs
    args: arguments to pass to the function calls
    kwargs: keyword arguments to pass to the function calls
    """
    # make sure we actually have a list of inputs...
    if not isinstance(maybe_input_or_inputs, list):
        inputs = [maybe_input_or_inputs]
    else:
        inputs = maybe_input_or_inputs
    # function is callable? just map it over the inputs
    if callable(maybe_fn_or_fn_list):
        return [maybe_fn_or_fn_list(arg, *args, *kwargs) for arg in inputs]
    # same number of inputs as functions: apply function i to arg i
    fn_list = maybe_fn_or_fn_list
    if len(fn_list) == len(inputs):
            return [fn_list[i](inputs[i], *args, **kwargs)
                    for i in range(len(fn_list))]
    # many inputs one function, apply function to inputs
    elif len(fn_list) is 1 and len(inputs) > 1:
            return [fn_list[0](inputs[i], *args, **kwargs)
                    for i in range(len(inputs))]
    # many functions on one input, apply functions independently
    elif len(fn_list) > 1 and len(inputs) == 1:
            return [fn_list[i](inputs[0], *args, **kwargs)
                    for i in range(len(fn_list))]
    else:
        raise Exception("map_enumerate fail",
                        maybe_fn_or_fn_list, maybe_input_or_inputs,
                        *args, **kwargs)
