import torch

from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

META_LITHO_REGISTRY = Registry("META_LITHO")
META_LITHO_REGISTRY.__doc__ = """
Registry for mata-litho, i.e. the whole litho.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_litho(cfg):
    """
    todo
    """
    meta_litho = cfg.LITHO_OPERATOR.NAME
    litho = META_LITHO_REGISTRY.get(meta_litho)(cfg)
    litho.to(torch.device(litho.device))
    _log_api_usage("modeling.litho." + meta_litho)
    return litho