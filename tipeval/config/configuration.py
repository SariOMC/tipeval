"""
Functions associated with the package configuration.
"""

from importlib.resources import path
import os
import yaml

from attrdict import AttrDict

from tipeval.core.typing import FilePath


def get_config_value() -> FilePath:
    """
    Get the location of the configuration file.

    :return: the file path of the config file
    """
    env_var_name = 'TIPEVAL_CONFIG'
    if env_var_name in os.environ:
        return os.environ[env_var_name]
    with path("tipeval", r'config.yaml') as f1:
        return f1


def load_configuration() -> AttrDict:
    """
    Read the configuration files associated with the package.

    This function merges the parameters specified in the config.yaml file
    and the parameters specified in configuration_intern.yaml which is
    in the tipeval.core module.

    :return: an AttrDict of the configuration saved in the yaml files
    """
    import tipeval.core
    with path(tipeval.core, 'configuration_intern.yaml') as f2:
        f1 = get_config_value()
        with open(f1, 'r') as c:
            config = AttrDict(yaml.safe_load(c))
        with open(f2, 'r') as c:
            config += AttrDict(yaml.safe_load(c))
    return config


"""The configuration dictionary (an attrdict)"""
configuration = load_configuration()
