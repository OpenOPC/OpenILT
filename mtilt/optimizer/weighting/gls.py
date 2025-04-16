import torch

from .Scaling import GradScalerBase, get_objectives_num
from .build import META_SCALER_REGISTRY
from mtilt.config import configurable, CfgNode

@META_SCALER_REGISTRY.register()
class GLS(GradScalerBase):
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
            "config": cfg,
        }

    @get_objectives_num
    def backward(self, losses=None, **kwargs):

        loss_data = torch.stack(losses)
        loss = torch.pow(loss_data.prod(), 1. / self.task_num)
        loss.backward()
        batch_weight = loss_data / (self.task_num * loss_data.prod())
        return batch_weight.detach().cpu().numpy()