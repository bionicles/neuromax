from nature import RaggedActuator, ImageActuator, Resizer, Classifier


def Actuator(agent, spec):
    if spec.rank is 3 and spec.shape[1] is None:
        return RaggedActuator(agent, spec)
    elif spec.rank is 4:
        return ImageActuator(agent)
    elif spec.format in ['discrete', 'onehot']:
        return Classifier(spec.shape)
    else:
        return Resizer(spec.shape)
