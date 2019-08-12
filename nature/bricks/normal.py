import tensorflow_probability as tfp

tfpl = tfp.layers

MIN_MIXTURE_COMPONENTS, MAX_MIXTURE_COMPONENTS = 2, 4
DISTRIBUTIONS = [
    "IndependentNormal", "MultivariateNormalTriL", "MixtureNormal"]


def get_normal(agent, brick_id, out_spec, distribution_name=None):
    if distribution_name is None:
        distribution_name = agent.pull_choices(
            f"{brick_id}_distribution_name", DISTRIBUTIONS)
    if distribution_name is "IndependentNormal":
        return tfpl.IndependentNormal(out_spec.shape)
    elif distribution_name is "MultivariateNormalTriL":
        return tfpl.MultivariateNormalTriL(out_spec.size)
    elif distribution_name is "MixtureNormal":
        n_components = agent.pull_numbers(
            f"{brick_id}_MixtureNormal_n_components",
            MIN_MIXTURE_COMPONENTS, MAX_MIXTURE_COMPONENTS)
        return tfpl.MixtureNormal(n_components, out_spec.shape)
    else:
        raise Exception(
            f"{brick_id} pulled unsupported distribution {distribution_name}")
