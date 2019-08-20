from tensorflow.keras import Input

from tools import make_uuid, log


def use_input(
        agent, id, input_or_spec,
        return_brick=True, shape_has_batch_dim=True, batch_size=None):
    id = make_uuid([id, "input"])
    if input_or_spec is None:
        input_or_spec = agent.code_spec

    log("use_input", id)
    log("we determine the shape of the input")
    if not isinstance(input_or_spec.shape, int) and shape_has_batch_dim:
        shape = input_or_spec.shape[1:]
    else:
        shape = input_or_spec.shape
    if batch_size is None:
        batch_size = agent.batch_size

    log("make the layer")
    layer = Input(shape, batch_size=batch_size)
    parts = dict(layer=layer, input_or_spec=input_or_spec)

    log("make call and brick")

    def call(*args):
        return layer
    return agent.build_brick(id, parts, call, "Bob", return_brick)
