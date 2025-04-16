import torch
import numpy as np
import random

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class PCB(GradScalerBase):
    r"""

      """

    @configurable
    def __init__(
            self,
            config: CfgNode,
            params: torch.Tensor,
    ):
        super().__init__(config, params)
        self.mask_num = config.LITHO_OPERATOR.PC_MASK_NUM

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        batch_weight = np.ones(self.task_num)
        self._compute_grad_dim()
        grads = self._compute_grad(loss_data, mode=self.mode)  # [task_num, grad_dim]

        if self.save_grad:
            if self.save_grad_order == 0:
                n_order_grads_show = grads
            else:
                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'pcb', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')

        if self.use_mask:

            keep_index = self._get_mask(grads, self.reference_index,
                                  mode=self.mask_mode,
                                  threshold=self.mask_threshold,
                                  ratio=self.mask_ratio)
            pc_grads = grads[keep_index].clone()
            self.task_num = len(keep_index)

            if not self.mask_pre_amplitude:
                grads = pc_grads.clone()
        else:
            pc_grads = grads.clone()

        visit = 0
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2) + 1e-8)
                    batch_weight[tn_j] -= (g_ij / (grads[tn_j].norm().pow(2) + 1e-8)).item()
                    visit +=1
                    if self.mask_num and (visit>=self.mask_num):
                        break
            if self.mask_num and (visit >= self.mask_num):
                break
        new_grads = pc_grads.sum(0)
        # new_grads.norm()
        # grads.sum(0).norm()

        alphas = [grad.norm() for grad in grads]
        alpha = max(alphas)
        new_grad_norm = new_grads / (new_grads.norm()+1e-7)
        new_grads = alpha * new_grad_norm

        if self.save_grad:
            self.save_grads(new_grads, 'pcb', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')
        self._reset_grad(new_grads)
        return sum(losses)