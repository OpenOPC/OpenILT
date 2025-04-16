from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

META_CONFIG_REGISTRY = Registry("META_CONFIG")
META_CONFIG_REGISTRY.__doc__ = """
Registry for mata-config, i.e. the whole config.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_config(cfg):
    """
    todo
    """
    meta_config = cfg.CONFIG.META_CONFIG
    config = META_CONFIG_REGISTRY.get(meta_config)(cfg)
    _log_api_usage("modeling.meta_config." + meta_config)
    return config