def sample_event(distribution):
    if isinstance(distribution.event_shape, int):
        event_length = 1
    else:
        len(distribution.event_shape)
    maybe_sample_or_event = distribution.sample(1)
    while len(maybe_sample_or_event.shape) > event_length:
        maybe_sample_or_event = maybe_sample_or_event[0]
    event = maybe_sample_or_event
    return event
