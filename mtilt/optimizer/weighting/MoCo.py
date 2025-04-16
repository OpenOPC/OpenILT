import torch
import torch.nn.functional as F

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class MoCo(GradScalerBase):
    r"""MoCo.

    This method is proposed in `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_ \
    and implemented based on the author' sharing code (Heshan Fernando: fernah@rpi.edu).

    Args:
        MoCo_beta (float, default=0.5): The learning rate of y.
        MoCo_beta_sigma (float, default=0.5): The decay rate of MoCo_beta.
        MoCo_gamma (float, default=0.1): The learning rate of lambd.
        MoCo_gamma_sigma (float, default=0.5): The decay rate of MoCo_gamma.
        MoCo_rho (float, default=0): The \ell_2 regularization parameter of lambda's update.

    .. warning::阿
            MoCo is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """

    @configurable
    def __init__(
            self,
            config: CfgNode,
            params: torch.Tensor,
            beta,
            beta_sigma,
            gamma,
            gamma_sigma,
            rho
    ):
        super().__init__(config, params)
        self.beta = beta
        self.beta_sigma = beta_sigma
        self.gamma = gamma
        self.gamma_sigma = gamma_sigma
        self.rho = rho

    @classmethod
    def from_config(cls, cfg):

        return {
            "config": cfg,
            "beta": cfg.ALGORITHM.MULTI_TASK_LEARNING.MOCO.BETA,
            "beta_sigma": cfg.ALGORITHM.MULTI_TASK_LEARNING.MOCO.BETA_SIGMA,
            "gamma": cfg.ALGORITHM.MULTI_TASK_LEARNING.MOCO.GAMMA,
            "gamma_sigma": cfg.ALGORITHM.MULTI_TASK_LEARNING.MOCO.GAMMA_SIGMA,
            "rho": cfg.ALGORITHM.MULTI_TASK_LEARNING.MOCO.RHO,
        }

    def init_param(self):
        self._compute_grad_dim()
        self.step = 0

    @get_objectives_num
    def backward(self, losses=None, **kwargs):
        loss_data = torch.stack(losses)
        self.step += 1

        if self.step == 1:
            self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)

        self._compute_grad_dim()
        grads = self._compute_grad(loss_data, mode=self.mode) # grads[0].data.cpu().numpy()

        if self.save_grad:
            n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'moco', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order + 1}_grads_{kwargs.get("iteration", None)}')
        with torch.no_grad():
            for tn in range(self.task_num):
                grads[tn] = grads[tn] / (grads[tn].norm() + 1e-8) * loss_data[tn] # min(grads.data.cpu().numpy()[1][torch.randint(10000,(1000,1))])
        self.y = self.y - (self.beta / self.step ** self.beta_sigma) * (self.y - grads) # torch.topk(grads[1], k=10000)[0].data.cpu().numpy()
        self.lambd = F.softmax(self.lambd - (self.gamma / self.step ** self.gamma_sigma) * (
                    self.y @ self.y.t() + self.rho * torch.eye(self.task_num).to(self.device)) @ self.lambd, -1)
        new_grads = self.y.t() @ self.lambd
        if self.save_grad:
            self.save_grads(new_grads, 'moco', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')
        self._reset_grad(new_grads) #new_grads.data.cpu().numpy()[torch.randint(10000,(1000,1))]   min(new_grads.data.cpu().numpy())
        return self.lambd.detach().cpu().numpy()
