from functools import partial

from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

META_SCALER_REGISTRY = Registry("META_SCALER")
META_SCALER_REGISTRY.__doc__ = """
Registry for mata-scaler, i.e. the whole scaler.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_scaler(cfg):
    """
    todo
    """
    meta_scaler = cfg.ALGORITHM.MULTI_TASK_LEARNING.SCALER
    scaler = partial(META_SCALER_REGISTRY.get(meta_scaler), cfg=cfg)
    _log_api_usage("modeling.meta_scaler." + meta_scaler)
    return scaler