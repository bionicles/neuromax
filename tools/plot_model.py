import tensorflow as tf

from tools import get_path, log

K = tf.keras

SHOW_SHAPES = False
SUMMARIZE = True
FOLDER = './pngs'


def plot_model(
        model, name, folder=FOLDER, extension="png", summarize=SUMMARIZE):
    """summarize and plot a keras model"""
    if summarize:
        model.summary()
    path = get_path(folder, name, extension)
    log(f"SCREENSHOT, {path}")
    K.utils.plot_model(model, path, show_shapes=SHOW_SHAPES)
