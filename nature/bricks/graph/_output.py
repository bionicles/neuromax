# graph_model.py - bion
import tensorflow as tf

from nature import Input, Coder, GraphBrick, ResidualBlock, Actuator, Linear

K = tf.keras
L = K.layers


def get_output(G, agent, id, task_model=False):
    node = G.node[id]
    if node["shape"] is "cylinder":
        return
    if node["output"] is not None:
        return node["output"]
    node_type = node["node_type"]
    if task_model and node_type is "input":
        node['input'] = input = output = Input(
            node["spec"]["shape"], batch_size=agent.batch_size)
        node['coder'] = coder = Coder(agent, node['spec'])
        output = coder(input)
    else:
        inputs = [get_output(G, agent, parent_id, task_model=task_model)
                  for parent_id in list(G.predecessors(id))]
        inputs = L.Concatenate(-1)(inputs) if len(inputs) > 1 else inputs[0]
        brick = None
        if node_type is 'brick':
            if task_model:
                brick = GraphBrick(agent, inputs=inputs)
            else:
                d_out = inputs.shape[-1]
                brick = ResidualBlock(
                    agent,
                    units_or_filters=d_out, layer_fn=Linear)
                output = brick(inputs)
                output = L.Concatenate(-1)([inputs, output])
        if node_type is "output":
            brick = Actuator(agent, node['spec'])
            output = brick(inputs)
        node["brick"] = brick
    node["output"] = output
    return output
