from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

META_ALGORITHM_REGISTRY = Registry("META_ALGORITHM")
META_ALGORITHM_REGISTRY.__doc__ = """
Registry for different (Multi-task learning) algorithms, i.e. the SGD optimizer.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_algorithm(cfg, scaler=None):
    """
    todo
    """
    meta_algorithm = cfg.ALGORITHM.NAME
    algorithm = META_ALGORITHM_REGISTRY.get(meta_algorithm)(cfg, scaler)
    _log_api_usage("solving.algorithm." + meta_algorithm)
    return algorithm