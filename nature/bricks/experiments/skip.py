def skip(x, brick):
    def call(x):
        return x + tf.reshape(brick(x), x.shape)
    return call
