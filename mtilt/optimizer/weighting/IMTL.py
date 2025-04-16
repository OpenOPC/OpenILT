import torch
import torch.nn as nn

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class IMTL(GradScalerBase):
    r"""Impartial Multi-task Learning (IMTL).

    This method is proposed in `Towards Impartial Multi-task Learning (ICLR 2021) <https://openreview.net/forum?id=IMPnRXEWpvr>`_ \
    and implemented by us.

    """

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
            "config": cfg
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        loss_scale = nn.Parameter(torch.tensor([0.0] * self.task_num, device=self.device))
        loss_data = loss_scale.exp() * loss_data - loss_scale
        grads = self._get_grads(loss_data, mode=self.mode)
        gard_backup = grads.detach().clone()

        if self.save_grad:
            if self.save_grad_order == 0:
                n_order_grads_show = grads
            else:
                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'imtl', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')
            # grads = n_order_grads_show[1,::]
        grads_unit = grads / torch.norm(grads, p=2, dim=-1, keepdim=True)

        D = grads[0:1].repeat(self.task_num - 1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(self.task_num - 1, 1) - grads_unit[1:]

        alpha = torch.matmul(torch.matmul(grads[0], U.t()), torch.inverse(torch.matmul(D, U.t())))
        alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)
        if self.save_grad:
            new_grads = sum([alpha[i] * grads[i] for i in range(self.task_num)])
            self.save_grads(new_grads, 'imtl', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')

        self._backward_new_grads(alpha, grads=gard_backup)
        return sum(losses)