import tensorflow as tf

from tools import get_path, log

K = tf.keras

SUMMARIZE = True
FOLDER = "."


def show_model(
        model, name, folder=FOLDER, extension="png", summarize=SUMMARIZE):
    """summarize and plot a keras model"""
    if summarize:
        model.summary()
    path = get_path(folder, name, extension)
    log(f"SCREENSHOT, {path}")
    K.utils.plot_model(model, path)
