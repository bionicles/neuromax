import tensorflow as tf


def get_onehot(item, list):
    """Return onehot encoding of item's position in a list
    Args:
        item: the object whose position will be 1
        list: the list of choices from which the item will be chosen

    Returns: a tensor of 0s except for a 1 in the position of item in list
    """
    return tf.convert_to_tensor([0 if x != list.index(item) else 1
                                 for x in range(len(list))])
