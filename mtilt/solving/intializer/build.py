import torch

from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

META_INITIALIZER_REGISTRY = Registry("META_INITIALIZER")
META_INITIALIZER_REGISTRY.__doc__ = """
Registry for different initializers, i.e. the level-set initializer.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_initializer(cfg):
    """
    todo
    """
    meta_initializer = cfg.INITIALIZER.NAME
    initializer = META_INITIALIZER_REGISTRY.get(meta_initializer)(cfg)
    _log_api_usage("solving.initializer." + meta_initializer)
    return initializer