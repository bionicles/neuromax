import tensorflow as tf

L = tf.keras.layers


def update(existing_aggregate, new_value):
    """
    for a new value, compute the new count, new mean, the new M2.
    mean accumulates the mean of the entire dataset
    M2 aggregates the squared distance from the mean
    count aggregates the number of samples seen so far
    """
    (count, mean, M2) = existing_aggregate
    count = count + 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)


def finalize(existing_aggregate):
    """retrieve the mean, variance and sample variance from an aggregate"""
    (count, mean, M2) = existing_aggregate
    (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        return (0., 0., 0.)
    else:
        return (mean, variance, sample_variance)


class OnlineStats(L.Layer):

    def __init__(self, x):
        super(OnlineStats, self).__init__()
        zeros = tf.zeros_like(x)
        self.aggregate = (0, zeros, zeros)

    def call(self, x):
        self.aggregate = update(self.aggregate, x)
        mean, variance, sample_variance = finalize(self.aggregate)
        return variance
