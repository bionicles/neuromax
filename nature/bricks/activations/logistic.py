import tensorflow as tf

from nature import L1L2

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


ones = K.initializers.ones

@tf.function
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
    return a + numerator / denominator


class Logistic(L.Layer):
    def __init__(self):
        super(Logistic, self).__init__()

    def build(self, input_shape):
        self.lower_asymptote = self.add_weight(
            initializer=tf.zeros, trainable=True)
        self.upper_asymptote_aka_carrying_capacity = self.add_weight(
            initializer=ones(), trainable=True)
        self.growth_rate = self.add_weight(
            initializer=ones(), trainable=True)
        self.is_related_to_value_y_zero = self.add_weight(
            initializer=ones(), trainable=True)
        self.is_added_to_exponential_term = self.add_weight(
            initializer=ones(), trainable=True)
        self.start_time = self.add_weight(
            initializer=ones(), trainable=True)
        self.location_of_max_growth = self.add_weight(
            initializer=ones(), trainable=True)
        self.built = True

    @tf.function
    def call(self, x):
        return generalized_logistic(
                x,
                a=self.lower_asymptote,
                k=self.upper_asymptote_aka_carrying_capacity,
                b=self.growth_rate,
                q=self.is_related_to_value_y_zero,
                c=self.is_added_to_exponential_term,
                m=self.start_time,
                v=self.location_of_max_growth)
