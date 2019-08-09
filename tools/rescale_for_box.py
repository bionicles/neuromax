import tensorflow as tf


def rescale_for_box(input, to_range, from_range):
    (from1, from2), (to1, to2) = from_range, to_range
    to_domain = to2 - to1
    from_domain = from2 - from1
    rescaled_input = to1 + ((input - from1) * to_domain / from_domain)
    tf.print(rescaled_input)
    return rescaled_input
