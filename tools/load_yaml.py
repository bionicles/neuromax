from tools import log
# https://pyyaml.org/wiki/PyYAMLDocumentation
import yaml

DEFAULT_PATH = './user_attrs.yaml'


def load_yaml(path=DEFAULT_PATH):
    log('loading a yaml file from', path)
    loaded = yaml.load(open(path, 'r'))
    log('loaded a yaml file:', loaded)
    return loaded
