def sample_event(distribution, sample_shape, batch_shape):
    sample = distribution.sample(sample_shape)
    len2discard = 1 if isinstance(sample_shape, int) else len(sample_shape)
    if isinstance(batch_shape, int):
        len2discard += 1
    else:
        len2discard += len(batch_shape)
    for _ in range(len2discard):
        sample = sample[0]
    return sample
