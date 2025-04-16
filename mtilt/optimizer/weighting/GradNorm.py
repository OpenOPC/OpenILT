import torch
import torch.nn.functional as F
import numpy as np

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class GradNorm(GradScalerBase):
    r"""Geometric Loss Strategy (GLS).

      This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
      and implemented by us.

      """

    @configurable
    def __init__(
            self,
            config: CfgNode,
            params: torch.Tensor,
            alpha: float,
    ):
        super().__init__(config, params)
        self.alpha = alpha

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg,
            "alpha": cfg.ALGORITHM.MULTI_TASK_LEARNING.GRADNORM.ALPHA,
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        # todo: fix bugs: how to record train_loss
        iteration = kwargs.get("iteration", None)
        loss_data = torch.stack(losses)
        if iteration >= 1:
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(loss_data, mode=self.mode)

            if self.save_grad:
                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order + 1)
                self.save_grads(n_order_grads_show, 'gradnorm', f'old/sample_{kwargs.get("num", None)}',
                                f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')

            G_per_loss = torch.norm(loss_scale.unsqueeze(1) * grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = torch.Tensor([loss_data[tn].item() / self.train_loss_buffer[tn, 0] for tn in range(self.task_num)]).to(
                self.device)
            r_i = L_i / L_i.mean()
            constant_term = (G * (r_i ** self.alpha)).detach()
            L_grad = (G_per_loss - constant_term).abs().sum(0)
            L_grad.backward()
            loss_weight = loss_scale.detach().clone()
            if self.save_grad:
                new_grads = sum([loss_weight[i] * grads[i] for i in range(self.task_num)])
                self.save_grads(new_grads, 'gradnorm', f'new/sample_{kwargs.get("num", None)}',
                                f'new_grads_{kwargs.get("iteration", None)}')
            self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            loss = torch.mul(loss_data, torch.ones_like(loss_data).to(self.device)).sum()
            loss.backward()

            if self.save_grad:
                grads_show = torch.zeros(len(self.grad_index), self.grad_dim).to(self.device)
                for index, param in enumerate(self._get_params()):
                    grads_show[index] = param.grad.data

                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order + 1)
                self.save_grads(n_order_grads_show, 'gradnorm', f'old/sample_{kwargs.get("num", None)}',
                                f'{self.save_grad_order + 1}_grads_0')

            return np.ones(self.task_num)