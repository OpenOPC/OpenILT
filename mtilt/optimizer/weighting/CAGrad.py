import torch
import  numpy as np

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

from scipy.optimize import minimize

@META_SCALER_REGISTRY.register()
class CAGrad(GradScalerBase):
    r"""Conflict-Averse Gradient descent (CAGrad).

    This method is proposed in `Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021) <https://openreview.net/forum?id=_61Qh8tULj_>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Cranial-XIX/CAGrad>`_.

    Args:
        calpha (float, default=0.5): A hyperparameter that controls the convergence rate.
        rescale ({0, 1, 2}, default=1): The type of the gradient rescaling.

    .. warning::
            CAGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """

    @configurable
    def __init__(
            self,
            config: CfgNode,
            params: torch.Tensor,
            calpha: float=0.5,
            rescale: int=1,
    ):
        super().__init__(config, params)

        self.calpha = calpha
        self.rescale = rescale

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg,
            "calpha": cfg.ALGORITHM.MULTI_TASK_LEARNING.CAGRAD.CALPHA,
            "rescale": cfg.ALGORITHM.MULTI_TASK_LEARNING.CAGRAD.RESCALE,
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        self._compute_grad_dim()
        grads = self._compute_grad(loss_data, mode=self.mode)

        if self.save_grad:
            if self.save_grad_order == 0:
                n_order_grads_show = grads
            else:
                n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'cagrad', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order+1}_grads_{kwargs.get("iteration", None)}')

        GG = torch.matmul(grads, grads.t()).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.task_num) / self.task_num
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (self.calpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1)) + c * np.sqrt(
                x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw
        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1 + self.calpha ** 2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError('No support rescale type {}'.format(self.rescale))
        if self.save_grad:
            self.save_grads(new_grads, 'cagrad', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')

        self._reset_grad(new_grads)

        return sum(losses)
