# import tensorflow as tf

from nature import Regresser, Classifier


def Actuator(AI, spec):
    # if spec.rank is 3 and spec.shape[1] is None:
    #     return RaggedActuator(agent, spec)
    # elif spec.rank is 4:
    #     return ImageActuator(agent)
    if spec.format in ['discrete', 'onehot']:
        return Classifier(AI, spec.shape)
    else:
        return Regresser(AI, spec.shape)