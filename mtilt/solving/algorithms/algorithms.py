import torch
import torch.nn as nn
import functools

from mtilt.config import configurable, CfgNode
from mtilt.utils.events import get_event_storage
from mtilt.optimizer import build_optimizer
from .build import META_ALGORITHM_REGISTRY


@META_ALGORITHM_REGISTRY.register()
class GeneralAlgorithm():

    @configurable
    def __init__(
            self,
            config: CfgNode,
            optimizer,
            objectives: dict,
            scaler,
            corners_list: list,
            mtl: bool=False,
            lr_schedule: bool = False,
            start_iter: int=30,
            schedule_step: int=5,
            lr_weight: float=0.5,
    ):
        self.config = config
        self._optimizer = optimizer
        self.base_lr = config.TRAINER.BASE_LR
        self.objectives = objectives
        self._scaler = scaler
        self.optimizer = None
        self.scheduler = lr_schedule
        self.start_iter = start_iter
        self.schedule_step = schedule_step
        self.lr_weight = lr_weight
        self.mtl = mtl
        self.corners_list = corners_list

    @classmethod
    def from_config(cls, cfg, scaler):

        return {"config": cfg,
                "mtl": cfg.ALGORITHM.MULTI_TASK_LEARNING.MTL,
                "optimizer": functools.partial(build_optimizer, cfg=cfg),
                "objectives": None,
                "scaler": scaler,
                "corners_list": cfg.ALGORITHM.CORNERS_LIST,
                "lr_schedule": cfg.ALGORITHM.LR_SCHEDULE.OPEN,
                "start_iter": cfg.ALGORITHM.LR_SCHEDULE.ITERATION,
                "schedule_step": cfg.ALGORITHM.LR_SCHEDULE.STEP_ITERATION,
                "lr_weight": cfg.ALGORITHM.LR_SCHEDULE.WEIGHT,
                }

    def bind_scaler(self, params):

        self.scaler = self._scaler(params=params)
        if hasattr(self.scaler, 'init_param'):
            self.scaler.init_param()

    def bind_optimizer(self, value):

        self.optimizer = self._optimizer(value=value)
        self.optimizer.param_groups[0]["lr"] = self.base_lr

    def __call__(self, loss_dict, params, **kwargs):

        self.optimizer.zero_grad()

        if self.scheduler and kwargs.get('iteration', None):
            if kwargs["iteration"] > self.start_iter and kwargs["iteration"] % self.schedule_step == 0:
                self.optimizer.param_groups[0]["lr"] *= self.lr_weight

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"litho_loss": loss_dict}
        else:
            losses = loss_dict["litho_loss"]
            if self.config.ALGORITHM.CORNERS:
                if self.corners_list:
                    # only consider some specific corners, like [0, 1, 3] or [2]
                    assert all(e < len(loss_dict["l2_loss"]) for e in self.corners_list), \
                        "invalid corner list index, which should be smaller than the corners total length"
                    losses = sum([loss_dict["l2_loss"][index]*len(loss_dict["l2_loss"]) for index in self.corners_list])
                else: # default=[], consider all corners
                    losses = sum(loss_dict["l2_loss"]) * len(loss_dict["l2_loss"])

        if not self.mtl:
            losses.backward()
        else:
            loss_dict.pop("litho_loss")
            for index, value in enumerate(loss_dict["l2_loss"]):
                loss_dict[f"l2_loss_corner{index}"] = value * len(loss_dict["l2_loss"])
            loss_dict.pop("l2_loss")
            if self.config.ALGORITHM.CORNERS:
                loss_dict.pop("pvb_l2_loss")
            losses = self.scaler.backward(losses=list(loss_dict.values()),
                                 iteration=kwargs.get('iteration', None),
                                 num=kwargs.get('num', None))

        self.optimizer.step()

        return losses
