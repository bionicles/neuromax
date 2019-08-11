import tensorflow as tf

from tools.get_path import get_path

K = tf.keras


def show_model(model, folder, name, extension, debug=True):
    """summarize and plot a keras model"""
    if debug:
        model.summary()
    K.utils.plot_model(model, get_path(folder, name, extension))
