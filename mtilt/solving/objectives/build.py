from mtilt.utils.logger import _log_api_usage
from mtilt.utils.registry import Registry

LOSSES_REGISTRY = Registry("LOSSES")
LOSSES_REGISTRY.__doc__ = """

""" # TODO


def build_litho_loss(cfg):
    """
    todo
    """
    litho_loss = cfg.LITHO_OPERATOR.LOSS.L2_LOSS
    loss_func = LOSSES_REGISTRY.get(litho_loss)(cfg)
    _log_api_usage("modeling.litho_loss." + litho_loss)
    return {"litho_l2_loss": loss_func}

def build_pvb_l2_loss(cfg):
    """
    todo
    """
    pvb_l2_loss = cfg.LITHO_OPERATOR.LOSS.PVB_L2_LOSS
    pvb_l2_loss_func = LOSSES_REGISTRY.get(pvb_l2_loss)(cfg)
    _log_api_usage("modeling.pvb_l2_loss." + pvb_l2_loss)
    return {"pvb_l2_loss": pvb_l2_loss_func}

def build_pvb_loss(cfg):
    """
    todo
    """
    pvb_loss = cfg.LITHO_OPERATOR.LOSS.PVB_LOSS
    pvb_loss_func = LOSSES_REGISTRY.get(pvb_loss)(cfg)
    _log_api_usage("modeling.pvb_loss." + pvb_loss)
    return {"pvb_loss": pvb_loss_func}


def build_extra_loss(cfg):
    """
    todo
    """
    extra_losses = cfg.LITHO_OPERATOR.LOSS.EXTRA_MULTI_LOSSES
    extra_loss_func = {}
    loss_weights = cfg.LITHO_OPERATOR.LOSS.EXTRA_MULTI_LOSSES_WEIGHTS
    mtl = cfg.ALGORITHM.MULTI_TASK_LEARNING
    assert len(loss_weights) == len(eval(extra_losses)), "the length of loss_weights should be the same with that of loss num"
    for index, (name, loss) in enumerate(eval(extra_losses).items()):
        extra_loss_func[name] = LOSSES_REGISTRY.get(loss)(loss_weights[index], mtl)
        _log_api_usage("modeling.extra_loss{}.".format(index) + name)
    return extra_loss_func

def build_losses(cfg):

    loss_funcs_dict = {}
    loss_funcs_dict.update(build_litho_loss(cfg))
    loss_funcs_dict.update(build_pvb_l2_loss(cfg))
    # loss_funcs_dict.update(build_pvb_loss(cfg))
    # loss_funcs_dict.update(build_extra_loss(cfg))

    return loss_funcs_dict

def build_objectives(cfg):

    objectives = cfg.LITHO_OPERATOR.LOSS.OBJECTIVES
    objectives_class = LOSSES_REGISTRY.get(objectives)(cfg)
    _log_api_usage("solving.objectives." + objectives)
    return objectives_class