import os.path

import torch.nn as nn
import torch
import matplotlib.pyplot as plt


from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode


@META_SCALER_REGISTRY.register()
class DB(GradScalerBase):

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
            self.save_grads(n_order_grads_show, 'DB', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')


        alphas = [grad.norm() for grad in grads]
        alpha = max(alphas)
        # alpha = sum(alphas)

        new_grad = alpha * sum([grad/(grad.norm()+1e-7) for grad in grads])
        self._reset_grad(new_grad) # new_grad.norm()

        if self.save_grad:
            self.save_grads(new_grad, 'DB', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')

        return sum(losses)

