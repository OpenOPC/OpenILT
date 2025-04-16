import torch.nn.functional as F
import torch

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class DWA(GradScalerBase):
    r"""Dynamic Weight Average (DWA).

    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.

    Args:
        T (float, default=2.0): The softmax temperature.

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
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        batch_weight = torch.ones_like(loss_data).to(self.device)
        loss = torch.mul(loss_data, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()