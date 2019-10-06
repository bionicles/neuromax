import tensorflow as tf
import numpy as np


def d(x, y, dist="ones", mask_ratio=0.04, same_mask=False, func="mse"):
    x = np.flatten(x)
    y = np.flatten(y)
    x_shape = x.shape[0]
    y_shape = y.shape[0]
    resizing = False

    if x_shape < y_shape:
        big_shape = y_shape
        size = x_shape
        resizing = "y"

    if y_shape < x_shape:
        big_shape = x_shape
        size = y_shape
        resizing = "x"

    y = np.expand_dims(y, 1)
    x = np.expand_dims(x, 1)

    # we resize the bigger array to the smaller size
    if resizing is not False:
        if dist == "uniform":
            resize = np.random.uniform(shape=(size, big_shape))
        if dist == "normal":
            resize = np.random.normal(shape=(size, big_shape))
        if dist == "ones":
            resize = np.ones(shape=(size, big_shape))
        if resizing == "y":
            y = np.dot(resize, y)
        if resizing == "x":
            x = np.dot(resize, x)

    # we mask x and y (dropout)
    if mask_ratio != 0:
        x_mask = np.random.choice(
            [0, 1], x.shape[0], p=[mask_ratio, 1-mask_ratio])
        if not same_mask:
            y_mask = np.random.choice(
                [0, 1], y.shape[0], p=[mask_ratio, 1-mask_ratio])
        if same_mask:
            y_mask = x_mask
        x = np.multiply(x, x_mask)
        y = np.multiply(y, y_mask)

    if func == "mpse":
        distance = tf.losses.mean_pairwise_squared_error(x, y)
    if func == "mse":
        distance = tf.losses.mean_squared_error(x, y)
    if func == "abs":
        distance = tf.losses.absolute_distance(x, y)
    return distance
