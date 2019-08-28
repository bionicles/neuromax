from nature import use_norm, use_fn


def add_classifier(prior_fn, spec):
    norm = use_norm()
    if spec.format is "discrete":
        classifier = use_fn(key='soft_argmax')
    elif spec.format is 'onehot':
        classifier = use_fn(key='softmax')

    def call(x):
        x = prior_fn(x)
        x = norm(x)
        return classifier(x)
    return call
