# https://towardsdatascience.com/4-awesome-things-you-can-do-with-keras-and-the-code-you-need-to-make-it-happen-9b591286e4e0
import tensorflow as tf

# usage:
# image_2 = resize_layer(scale=2)(image, method="bilinear")


def tf_int_round(num):
    return tf.cast(tf.round(num), dtype=tf.int32)


class ResizeImage(tf.keras.layers.Layer):
    # Initialize variables
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super(ResizeImage, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResizeImage, self).build(input_shape)

    def get_height_width(self, x, y):
        height = tf_int_round(tf.cast(y, dtype=tf.float32) * self.scale)
        width = tf_int_round(tf.cast(x, dtype=tf.float32) * self.scale)
        return height, width

    def call(self, x, method="bicubic"):
        shape = tf.shape(x)
        height, width = self.get_height_width(shape[1], shape[2])
        if method == "bilinear":
            return tf.image.resize_bilinear(x, size=(height, width))
        elif method == "bicubic":
            return tf.image.resize_bicubic(x, size=(height, width))
        elif method == "nearest":
            return tf.image.resize_nearest_neighbor(x, size=(height, width))

    def get_output_shape_for(self, input_shape):
        height, width = self.get_height_width(input_shape[1], input_shape[2])
        return (self.input_shape[0], height, width, input_shape[3])
