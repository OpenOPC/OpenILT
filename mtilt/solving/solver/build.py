from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

META_SOLVER_REGISTRY = Registry("SOLVER")
META_SOLVER_REGISTRY.__doc__ = """
Registry for meta-solver.

The registered object will be called with `obj(cfg)`.
"""


def build_solver(cfg):
    """
    todo
    """

    # todo: change the configurations
    meta_solver = cfg.SOLVER.META_SOLVER
    solver = META_SOLVER_REGISTRY.get(meta_solver)(cfg)
    _log_api_usage("solving.solver." + meta_solver)
    return solver