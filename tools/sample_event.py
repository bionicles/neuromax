from .log import log


def sample_event(distribution):
    if isinstance(distribution.event_shape, int):
        event_length = 1
    else:
        event_length = len(distribution.event_shape)
    maybe_sample_or_event = distribution.sample(1)
    log("maybe_sample_or_event", maybe_sample_or_event.shape[-1], color="blue")
    try:
        while len(maybe_sample_or_event.shape) > event_length:
            maybe_sample_or_event = maybe_sample_or_event[0]
    except Exception as e:
        print(e)
    event = maybe_sample_or_event
    return event
