import tensorflow_probability as tfp
import tensorflow as tf
import random

K = tf.keras
L = K.layers
tfpl = tfp.layers

INPUT_SHAPES = [(1, 1), (1, 3), (1, 420, 300), (1, 128, 128, 6)]
CODE_SHAPE = (42, 16)
MIN_MIXTURE_COMPONENTS, MAX_MIXTURE_COMPONENTS = 2, 4
DISTRIBUTIONS = ["IndependentNormal"]


def get_normal(distribution_name, size, shape):
    print(f"{distribution_name} distribution")
    if distribution_name is "IndependentNormal":
        cls = tfpl.IndependentNormal
        params_size = cls.params_size(shape)
        instance = cls(shape)
    elif distribution_name is "MultivariateNormalTriL":
        cls = tfpl.MultivariateNormalTriL
        params_size = cls.params_size(size)
        instance = cls(size)
    elif distribution_name is "MixtureNormal":
        cls = tfpl.MixtureNormal
        n_components = random.randint(MIN_MIXTURE_COMPONENTS, MAX_MIXTURE_COMPONENTS)
        print(f"{n_components} components")
        params_size = cls.params_size(n_components, shape)
        instance = cls(n_components, shape)
    return cls, params_size, instance


def main():
    for INPUT_SHAPE in INPUT_SHAPES:
        code_size = CODE_SHAPE[0] * CODE_SHAPE[1]
        codes, samples = [], []
        for i in range(3):
            cls, params_size, instance = get_normal("IndependentNormal", code_size, CODE_SHAPE)
            sensor = K.Sequential([
                K.Input(INPUT_SHAPE),
                L.Flatten(),
                L.Dense(params_size),
                instance
            ])
            random_input = tf.random.normal(INPUT_SHAPE)
            code = sensor(random_input)
            codes.append(code)
            print("params", params_size)
            print("code shape", code.shape)
            sample = code.sample()
            print("sample shape", sample.shape)
            samples.append(sample)
        code = tf.concat(codes, 1)
        print("concat codes", code.shape)


main()
