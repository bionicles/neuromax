from nature import Norm, Fn, Multiply, Resizer, Brick


def Classifier(agent, spec):
    m1 = Multiply()
    m2 = Multiply()
    n1 = Norm()
    n2 = Norm()
    n3 = Norm()
    n4 = Norm()
    resizer = Resizer(agent, spec.shape)
    if spec.format is "discrete":
        classifier = Fn(key='soft_argmax')
    elif spec.format is 'onehot':
        classifier = Fn(key='softmax')

    def call(self, x):
        x = m1(n1(x))
        x = m2(n2(x))
        x = resizer(n3(x))
        return classifier(n4(x))
    return Brick(m1, m2, n1, n2, n3, n4, resizer, classifier, call, agent)
