import torch
import numpy as np
import random

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class PCGrad(GradScalerBase):
    r"""Geometric Loss Strategy (GLS).

      This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
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
        batch_weight = np.ones(self.task_num)
        self._compute_grad_dim()
        grads = self._compute_grad(loss_data, mode=self.mode)  # [task_num, grad_dim]

        if self.save_grad:
            if self.save_grad_order == 0:
                n_order_grads_show = grads
            else:
                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'pcgrad', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2) + 1e-8)
                    batch_weight[tn_j] -= (g_ij / (grads[tn_j].norm().pow(2) + 1e-8)).item()
        new_grads = pc_grads.sum(0)
        if self.save_grad:
            self.save_grads(new_grads, 'pcgrad', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')
        self._reset_grad(new_grads)
        return sum(losses)