import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtilt.config import configurable, CfgNode
from .build import LOSSES_REGISTRY, build_losses

@LOSSES_REGISTRY.register()
class MSE_Loss(nn.Module):
    # todo
    def __init__(self, weight=1.0, mtl=False, **kwargs):
        super().__init__()

        if mtl:
            self.weight = 1.0
        else:
            self.weight = weight

    def forward(self, printedNom, printedMax, printedMin, target):

        return self.weight * F.mse_loss(printedNom, target, reduction="sum")


@LOSSES_REGISTRY.register()
class L1_Loss(nn.Module):
    # todo
    def __init__(self, weight=1.0, mtl=False, **kwargs):
        super().__init__()

        if mtl:
            self.weight = 1.0
        else:
            self.weight = weight

    def forward(self, **kwargs):
        loss = {}

        return self.weight * loss


@LOSSES_REGISTRY.register()
class Smooth_L1_Loss(nn.Module):
    # todo
    def __init__(self, weight=1.0, mtl=False, **kwargs):
        super().__init__()

        if mtl:
            self.weight = 1.0
        else:
            self.weight = weight

    def forward(self, **kwargs):
        loss = {}

        return self.weight * loss

@LOSSES_REGISTRY.register()
class BasicObjecitvesDict():

    @configurable
    def __init__(
            self,
            cfg,
            weight_l2: float,
            weight_pvb_l2: float,
            weight_pvband: float,
            device,
            dtype,
            curv,
    ):
        self.config = cfg
        self.loss_funcs = {}
        self.loss_funcs.update(build_losses(cfg))

        self.weight_l2 = weight_l2
        self.weight_pvb_l2 = weight_pvb_l2
        self.weight_pvband = weight_pvband
        self.curv = curv

        self.device = device
        self.dtype = dtype


    @classmethod
    def from_config(cls, cfg):

        return {"cfg": cfg,
                "weight_l2": cfg.SOLVER.WEIGHT_L2,
                "weight_pvb_l2": cfg.SOLVER.WEIGHT_PVBL2,
                "weight_pvband": cfg.SOLVER.WEIGHT_PV_Band,
                "device": cfg.LITHO_OPERATOR.DEVICE,
                "dtype": cfg.REALTYPE,
                "curv": cfg.LITHO_OPERATOR.LOSS.CURV,
                }

    def add_loss_func(self, key, loss_func):
        if key in self.loss_funcs:
            print(f"Warning: Key '{key}' already exists. Overwriting the existing value.")
        self.loss_funcs[key] = loss_func

    def get(self, key):
        return self.loss_funcs.get(key, None)

    def remove(self, key):
        if key in self.loss_funcs:
            del self.loss_funcs[key]
        else:
            print(f"Warning: Key '{key}' not found.")

    @property
    def keys(self):
        return list(self.loss_funcs.keys())

    def losses(self, mask, printedNoms, printedMax, printedMin, target):
        loss_values = {}
        loss_values.update(self.tmp_losses(mask, printedNoms, printedMax, printedMin, target))
        # for key, loss_func in self.loss_funcs.items():
        #     loss = MSE_Loss()(printedNom, printedMax, printedMin, target)
        #     # loss = loss_func(printedNom, printedMax, printedMin, target)
        #     loss_values[key] = loss
        return loss_values

    def update(self, loss_func_dict):
        for key, loss_func in loss_func_dict.items():
            self.add_loss_func(key, loss_func)

    # # todo: need to further modify such pattern
    def tmp_losses(
            self,
            mask,
            printedNoms,
            printedMax,
            printedMin,
            target,
            curv=None,
    ):

        l2loss = [F.mse_loss(printedNom, target, reduction="sum") for printedNom in printedNoms]
        pvb_l2_loss = F.mse_loss(printedMax, target, reduction="sum") + \
                      F.mse_loss(printedMin, target, reduction="sum")
        pvb_loss = F.mse_loss(printedMax, printedMin, reduction="sum")
        litho_loss = self.weight_l2*sum(l2loss)/len(l2loss) + self.weight_pvb_l2 * pvb_l2_loss + self.weight_pvband * pvb_loss
        # litho_loss = self.weight_l2 * l2loss[4] + self.weight_pvb_l2 * pvb_l2_loss + self.weight_pvband * pvb_loss

        if curv is not None:
            kernelCurv = torch.tensor(
                [[-1.0 / 16, 5.0 / 16, -1.0 / 16], [5.0 / 16, -1.0, 5.0 / 16], [-1.0 / 16, 5.0 / 16, -1.0 / 16]],
                dtype=self.dtype, device=self.device)
            curvature = F.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
            losscurv = F.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
            litho_loss += curv * losscurv

        return {"l2_loss": [self.weight_l2 * l2 / len(l2loss) for l2 in l2loss],
                "pvb_l2_loss": self.weight_pvb_l2 * pvb_l2_loss,
                "litho_loss": litho_loss}