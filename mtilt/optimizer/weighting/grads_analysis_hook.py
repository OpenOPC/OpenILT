import os.path

import torch.nn as nn
import torch
import matplotlib.pyplot as plt


from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode


@META_SCALER_REGISTRY.register()
class Analysis(GradScalerBase):

    @configurable
    def __init__(
            self,
            config: CfgNode,
            params: torch.Tensor,
    ):
        super().__init__(config, params)

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg,
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        grads = self._get_grads(loss_data, mode=self.mode)

        if self.save_grad:
            if self.save_grad_order == 0:
                n_order_grads_show = grads
            else:
                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'base', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')
            new_grads = n_order_grads_show if not self.cfg.ALGORITHM.CORNERS_LIST \
                            else n_order_grads_show[self.cfg.ALGORITHM.CORNERS_LIST]
            self.save_grads(new_grads, 'base', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')

        if self.cfg.ALGORITHM.CORNERS_LIST:
            self.task_num = len(self.cfg.ALGORITHM.CORNERS_LIST)
            grads = grads[self.cfg.ALGORITHM.CORNERS_LIST]
        alpha = torch.ones(self.task_num, device=grads.device)
        self._backward_new_grads(alpha, grads=grads)

        return sum(losses)

