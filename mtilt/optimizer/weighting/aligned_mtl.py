import torch

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class Aligned_MTL(GradScalerBase):
    r"""Aligned-MTL.

    This method is proposed in `Independent Component Alignment for Multi-Task Learning (CVPR 2023) <https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/SamsungLabs/MTL>`_.

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
            "config": cfg,
            "device": cfg.LITHO_OPERATOR.DEVICE,
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        grads = self._get_grads(loss_data, mode=self.mode)
        if self.save_grad:
            n_order_grads_show = self._compute_n_order_grad(loss_data, self.save_grad_order+1)
            self.save_grads(n_order_grads_show, 'alignmtl', f'old/sample_{kwargs.get("num", None)}',
                            f'{self.save_grad_order+1}_grads_{kwargs.get("iteration", None)}')

        M = torch.matmul(grads, grads.t())  # [num_tasks, num_tasks]
        lmbda, V = torch.symeig(M, eigenvectors=True)
        tol = (
                torch.max(lmbda)
                * max(M.shape[-2:])
                * torch.finfo().eps
        )
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
        alpha = B.sum(0)
        if self.save_grad:
            new_grads = sum([alpha[i] * grads[i] for i in range(self.task_num)])
            self.save_grads(new_grads, 'alignmtl', f'new/sample_{kwargs.get("num", None)}',
                            f'new_grads_{kwargs.get("iteration", None)}')

        self._backward_new_grads(alpha, grads=grads)

        return alpha.detach().cpu().numpy()
