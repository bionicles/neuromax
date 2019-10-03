import tensorflow as tf
import nature as N
L = tf.keras.layers


class Knowm(L.Layer):

    def __init__(self, AI, components):
        super().__init__()
        self.components = components
        self.n = len(components)
        self.ai = AI

    def build(self, shapes):
        self.classifier = N.Actuator(self.ai, (shapes[0], self.n))
        super().build(shapes)

    @tf.function
    def call(self, x):
        if tf.is_tensor(x):
            x = [x for _ in range(self.n)]
        component_outputs = []
        for component in self.components:
            component_outputs.append(component(x))
        weights = self.classifier(x)
        instance_weights = tf.split(weights, self.batch, axis=0)
        outputs = []
        for instance_weight in instance_weights:
            instance_outputs = []
            component_weights = tf.split(instance_weight, self.n, axis=1)
            for output, weight in zip(component_outputs, component_weights):
                instance_outputs.append(output * weight)
            instance_output = self.add(instance_outputs)
            outputs.append(instance_output)
        return tf.concat(outputs, axis=0)
