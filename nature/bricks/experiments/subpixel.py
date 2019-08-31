from __future__ import division
import tensorflow as tf

K, B, L = tf.keras, tf.keras.backend, tf.keras.layers

"""
https://github.com/atriumlts/subpixel/blob/master/keras_subpixel.py

Subpixel Layer as a child class of Conv2D. This layer accepts all normal
arguments, with the exception of dilation_rate(). The argument r indicates
the upsampling factor, which is applied to the normal output of Conv2D.
The output of this layer will have the same number of channels as the
indicated filter field, and thus works for grayscale, color, or as a a
hidden layer.
Arguments:
    *see Keras Docs for Conv2D args, note dilation_rate() is removed*
    r: upscaling factor, which is applied to the output of normal Conv2D
"""


class Subpixel(L.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,  # resize_factor
                 padding='valid',
                 data_format=None,
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = B.shape(I)[0]  # Handle undefined batch dim
        X = B.reshape(I, [bsize, a, b, c / (r * r), r, r])  # bsize, a, b, c/(r*r), r, r
        X = B.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        # Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:, i, :, :, :, :] for i in range(a)]  # a, [bsize, b, r, r, c/(r*r)
        X = B.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:, i, :, :, :] for i in range(b)]  # b, [bsize, r, r, c/(r*r)
        X = B.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r * unshifted[1], self.r * unshifted[2],
                unshifted[3] / (self.r * self.r))

    def get_config(self):
        config = super(L.Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] /= self.r * self.r
        config['r'] = self.r
        return config


class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=1):
        self.initializer = initializer
        self.scale = scale

    def __call__(self, shape, dtype, partition_info=None):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)
        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype, partition_info)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(
                x, size=(shape[0] * self.scale, shape[1] * self.scale))
        x = tf.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        return x


# def SubpixelConv2D(input_shape, scale=4):
#     """
#     Keras layer to do subpixel convolution.
#     NOTE: Tensorflow backend only. Uses tf.depth_to_space
#
#     out = SubpixelConv2D(input_shape, scale=scale)(x)
#
#     Ref:
#         [1] Real-Time Single Image and Video Super-Resolution
#         Using an Efficient Sub-Pixel Convolutional Neural Network
#         Shi et al. https://arxiv.org/abs/1609.05158
#         https://arxiv.org/pdf/1707.02937.pdf
#     :param input_shape: tensor shape, (batch, height, width, channel)
#     :param scale: upsampling scale. Default=4
#     :return:
#     """
#     def subpixel_shape(input_shape):
#         dims = [input_shape[0],
#                 input_shape[1] * scale,
#                 input_shape[2] * scale,
#                 int(input_shape[3] / (scale ** 2))]
#         output_shape = tuple(dims)
#         return output_shape
#
#     def subpixel(x):
#         return tf.depth_to_space(x, scale)
#
#     return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')


# https://github.com/atriumlts/subpixel/issues/33#issuecomment-446352163
# def SubpixelReshuffleLayer(inputs):
#     batch_size, rows, cols, in_channels = inputs.get_shape().as_list()
#     kernel_filter_size = 2
#     out_channels = int(in_channels // 4)
#     kernel_shape = [kernel_filter_size, kernel_filter_size, out_channels, in_channels]
#     kernel = np.zeros(kernel_shape, np.float32)
#
#     # Build the kernel so that a 4 pixel cluster has each pixel come from a separate channel.
#     for c in range(0, out_channels):
#         i = 0
#         for x, y in itertools.product(range(2), repeat=2):
#             kernel[y, x, c, c * 4 + i] = 1
#             i += 1
#
#     new_rows, new_cols = int(rows * 2), int(cols * 2)
#     new_shape = [batch_size, new_rows, new_cols, out_channels]
#     tf_shape = tf.stack(new_shape)
#     strides_shape = [1, 2, 2, 1]
#
#     out = tf.nn.conv2d_transpose(inputs, kernel, tf_shape, strides_shape, padding='VALID')
#
#     return out
