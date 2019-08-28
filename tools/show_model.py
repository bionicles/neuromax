import tensorflow as tf

from tools.get_path import get_path

K = tf.keras

SUMMARIZE = False
FOLDER = "."


def show_model(
        model, name, folder=FOLDER, extension="png", summarize=SUMMARIZE):
    """summarize and plot a keras model"""
    if summarize:
        model.summary()
    K.utils.plot_model(model, get_path(folder, name, extension))
