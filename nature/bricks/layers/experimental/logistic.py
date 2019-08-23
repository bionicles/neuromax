import tensorflow as tf

K = tf.keras
B, L = K.backend, K.layers

LOWER_ASYMPTOTE = 0
UPPER_ASYMPTOTE_AKA_CARRYING_CAPACITY = 1.
GROWTH_RATE = 1.
LOCATION_OF_MAX_GROWTH = 1.
START_TIME = 0.
COEFFICIENT_OF_EXPONENTIAL_TERM = 1.
IS_RELATED_TO_VALUE_Y_ZERO = 1.
IS_ADDED_TO_EXPONENTIAL_TERM = 1.


def generalized_logistic(
        x,
        a=LOWER_ASYMPTOTE,
        k=UPPER_ASYMPTOTE_AKA_CARRYING_CAPACITY,
        b=GROWTH_RATE,
        q=IS_RELATED_TO_VALUE_Y_ZERO,
        c=IS_ADDED_TO_EXPONENTIAL_TERM,
        m=START_TIME,
        v=LOCATION_OF_MAX_GROWTH,
        ):
    numerator = k - a
    exponential_term = B.exp(-b * (x - m))
    denominator = (c + q * exponential_term ** (1/v))
    y = a + numerator / denominator
    return y


class GeneralizedLogistic(L.Layer):
    def __init__(self):
        super(GeneralizedLogistic, self).__init__()

    def build(self, input_shape):
        self.lower_asymptote = tf.Variable(
            0., trainable=True)
        self.upper_asymptote_aka_carrying_capacity = tf.Variable(
            1., trainable=True)
        self.growth_rate = tf.Variable(
            1., trainable=True)
        self.is_related_to_value_y_zero = tf.Variable(
            1., trainable=True)
        self.is_added_to_exponential_term = tf.Variable(
            1., trainable=True)
        self.start_time = tf.Variable(
            1., trainable=True)
        self.location_of_max_growth = tf.Variable(
            1., trainable=True)

    def call(self, x):
        return generalized_logistic(
                x,
                a=self.lower_asymptote,
                k=self.upper_asymptote_aka_carrying_capacity,
                b=self.growth_rate,
                q=self.is_related_to_value_y_zero,
                c=self.is_added_to_exponential_term,
                m=self.start_time,
                v=self.location_of_max_growth,
                )
