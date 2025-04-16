import itertools
import logging
from itertools import chain, product
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtilt.config import configurable, CfgNode
from mtilt.utils.registry import Registry
from .build import META_LITHO_REGISTRY
from mtilt.config import get_cfg
from .generalized_litho_operator import Binarize, build_kernel
from ..objectives import build_litho_loss, \
    build_pvb_l2_loss, build_pvb_loss, build_extra_loss

KERNEL_REGISTRY = Registry("KERNEL")
KERNEL_REGISTRY.__doc__ = """

""" # todo

logger = logging.getLogger(__name__)


def _maskFloat(mask, dose):
    return (dose * mask).to(eval(get_cfg().COMPLEXTYPE))

def _kernelMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    output = None
    if kernel.device != maskFFT.device:
        kernel = kernel.to(maskFFT.device)
    if len(maskFFT.shape) == 3:
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :knxh+1, :knyh+1] = maskFFT[:, :knxh+1, :knyh+1] * kernel[:kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :knxh+1, -knyh:] = maskFFT[:, :knxh+1, -knyh:] * kernel[:kernelNum, -(knxh+1):, :knyh]
        output[:, -knxh:, :knyh+1] = maskFFT[:, -knxh:, :knyh+1] * kernel[:kernelNum, :knxh, -(knyh+1):]
        output[:, -knxh:, -knyh:] = maskFFT[:, -knxh:, -knyh:] * kernel[:kernelNum, :knxh, :knyh]
    else:
        assert len(maskFFT.shape) == 4, f"[_kernelMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :knxh+1, :knyh+1] = maskFFT[:, :, :knxh+1, :knyh+1] * kernel[None, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :knxh+1, -knyh:]  = maskFFT[:, :, :knxh+1, -knyh:]  * kernel[None, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, -knxh:, :knyh+1]  = maskFFT[:, :, -knxh:, :knyh+1]  * kernel[None, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, -knxh:, -knyh:]   = maskFFT[:, :, -knxh:, -knyh:]   * kernel[None, :kernelNum, :knxh, :knyh]
    return output

def _computeImageMatrix(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device:
        scale = scale.to(cmask.device)
    assert len(cmask.shape) in [3, 4], f"[_computeImageMask]: Invalid shape: {cmask.shape}"
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp

def _computeImageMask(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device:
        scale = scale.to(cmask.device)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp

def _convMatrix(cmask, dose, kernel, scale, kernelNum):
    image = _computeImageMatrix(cmask, kernel, scale, kernelNum)
    return image
def _convMask(mask, dose, kernel, scale, kernelNum):
    cmask = _maskFloat(mask, dose)
    image = _computeImageMask(cmask, kernel, scale, kernelNum)
    return image

class _LithoSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad):
        ctx.saved = (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad)
        tmp = _convMask(mask, dose, kernel, scale, kernelNum)
        if len(mask.shape) == 2:
            scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
            return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
        else:
            assert len(mask.shape) == 3, f"[_LithoSim.forward]: Invalid shape: {mask.shape}"
            scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)
    @staticmethod
    def backward(ctx, grad):
        (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad) = ctx.saved
        cpx0 = torch.mul(_convMask(mask, dose, kernelGradCT, scaleGradCT, kernelNumGradCT), grad.unsqueeze(len(grad.shape) - 2))
        cpx1 = _convMatrix(cpx0, dose, kernelGrad, scaleGrad, kernelNumGrad)
        cpx2 = torch.mul(_convMask(mask, dose, kernelGrad, scaleGrad, kernelNumGrad), grad.unsqueeze(len(grad.shape) - 2))
        cpx3 = _convMatrix(cpx2, dose, kernelGradCT, scaleGradCT, kernelNumGradCT)
        cpx4 = cpx1 + cpx3
        if len(mask.shape) == 2:
            scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
            cpx4 = torch.sum(scale * cpx4, dim=0)
        else:
            assert len(mask.shape) == 3, f"[_LithoSim.forward]: Invalid shape: {mask.shape}"
            scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            cpx4 = torch.sum(scale * cpx4, dim=1)

        return cpx4.real, None, None, None, None, None, None, None, None, None, None

@META_LITHO_REGISTRY.register()
class ExactLithoMultiSim(nn.Module):
    """
        todo
        """

    @configurable
    def __init__(
            self,
            *,
            config: CfgNode,
            filter: torch.Tensor,
            kernels: dict,
            required: list,
            dose_list: list,
            sigmoid_steepness: float,
            print_steepness: float,
            target_threshold: float,
            weight_pvb_l2: float,
            weight_pvband: float,
            kernel_num: int,
            l2_loss,
            pvb_l2_loss,
            pvb_loss,
            curv=None,
            corners_list,
            corners_forward: bool = False,
            device: str,
            dtype,
            # int_fields: list,
            # float_fields: list,
            multi_task: bool = False,
            initializer_type: str="pixel",
            mask: bool=False,
            mask_ratio: float=0.8,

    ):
        super().__init__()
        self.cfg = config

        self.filter = filter
        self.kernels = kernels
        self.kernel_num = kernel_num


        # defocus kernel
        self.focus_kernel = (self.kernels["focus"].kernels,
                                    self.kernels["focus"].scales,
                                    self.kernel_num,
                                    self.kernels["ct focus"].kernels,
                                    self.kernels["ct focus"].scales,
                                   self.kernel_num,
                                    self.kernels["focus"].kernels,
                                    self.kernels["focus"].scales,
                                   self.kernel_num)
        self.defocus_kernel = (self.kernels["defocus"].kernels,
                                   self.kernels["defocus"].scales,
                                   self.kernel_num,
                                   self.kernels["ct defocus"].kernels,
                                   self.kernels["ct defocus"].scales,
                                   self.kernel_num,
                                   self.kernels["defocus"].kernels,
                                   self.kernels["defocus"].scales,
                                   self.kernel_num)

        # dose
        self.dose = dose_list
        self.dose_nom = dose_list[len(dose_list)//2]
        self.dose_min = dose_list[0]
        self.dose_max = dose_list[-1]

        # mask
        self.mask = mask
        self.mask_ratio = mask_ratio
        self.corners_config = list(itertools.product([self.defocus_kernel, self.focus_kernel],
                                         self.dose))

        self.required = required

        self.sigmoid_steepness = sigmoid_steepness
        self.print_steepness = print_steepness
        self.target_threshold = target_threshold

        self.weight_pvb_l2 = weight_pvb_l2
        self.weight_pvband = weight_pvband

        self.curv = curv
        self.l2_loss_func = l2_loss
        self.pvb_l2_loss_func = pvb_l2_loss
        self.pvb_loss_func = pvb_loss

        self.corners_forward = corners_forward
        self.corners_list = corners_list

        self.device = device
        self.dtype = dtype

        # self.int_fields = int_fields
        # self.float_fields = float_fields

        # multi task/objective learning
        self.multi_task = multi_task
        if multi_task:
            self.extra_loss_funcs = build_extra_loss(config)

        assert initializer_type in ["pixel", "levelset"], "please check your initializer type!"
        logger.info(f"**ATTENTION!! You are using {initializer_type} type initializer.**")
        self.initializer_type = initializer_type
        if initializer_type == "levelset":
            self.binarizer = Binarize()

        self.training = True

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        if value is not None:
            self._filter = value
        else:
            raise ValueError("invalid value for a torch.Tensor format data.")

    @classmethod
    def from_config(cls, cfg):

        filter = torch.zeros([cfg.DESIGN.TILE_SIZE_X,
                              cfg.DESIGN.TILE_SIZE_Y],
                             dtype=eval(cfg.REALTYPE),
                             device=cfg.LITHO_OPERATOR.DEVICE)
        filter[cfg.DESIGN.OFF_SET_X:cfg.DESIGN.OFF_SET_X + cfg.DESIGN.ILT_SIZE_X, \
               cfg.DESIGN.OFF_SET_Y:cfg.DESIGN.OFF_SET_Y + cfg.DESIGN.ILT_SIZE_Y] = 1

        return {"config": cfg,
                "filter": filter,
                "kernels": build_kernel(cfg),
                "required": cfg.LITHO_OPERATOR.REQUIRED,
                "dose_list": cfg.LITHO_OPERATOR.DOSE_LIST,
                "sigmoid_steepness": cfg.SOLVER.SIGMOID_STEEPNESS,
                "print_steepness": cfg.LITHO_OPERATOR.PRINT_STEEPNESS,
                "target_threshold": cfg.LITHO_OPERATOR.TARGET_SHRESHOLD,
                "weight_pvb_l2": cfg.SOLVER.WEIGHT_PVBL2,
                "weight_pvband": cfg.SOLVER.WEIGHT_PV_Band,
                "kernel_num": cfg.LITHO_OPERATOR.KERNEL_NUM,
                "curv": cfg.LITHO_OPERATOR.LOSS.CURV,
                "l2_loss": build_litho_loss(cfg),
                "pvb_l2_loss": build_pvb_l2_loss(cfg),
                "pvb_loss": build_pvb_loss(cfg),
                "corners_forward": cfg.SOLVER.CORNERS_FORWARD,
                "device": cfg.LITHO_OPERATOR.DEVICE,
                "dtype": cfg.REALTYPE,
                "multi_task": cfg.ALGORITHM.MULTI_TASK_LEARNING.MTL,
                "initializer_type": cfg.INITIALIZER.TYPE,
                "corners_list": cfg.ALGORITHM.CORNERS_LIST,
                "mask": cfg.LITHO_OPERATOR.MASK,
                "mask_ratio": cfg.LITHO_OPERATOR.MASK_RATIO,
                }

    def bind_backup(self, backup):
        self.backup = backup

    def _pre_process(self, params):

        if self.initializer_type == "pixel":
            mask = torch.sigmoid(self.sigmoid_steepness * params) * self.filter
            mask += torch.sigmoid(self.sigmoid_steepness * self.backup) * (1.0 - self.filter)
        else:
            mask = params * self.filter + self.backup * (1.0 - self.filter)
            mask = self.binarizer(mask)
        return mask  # mask[mask>0]

    def corner_forward(self, mask, dose, *focus_kernel):

        aerial_image = _LithoSim.apply(mask, dose, *focus_kernel)
        printed_image = torch.sigmoid(self.print_steepness * (aerial_image - self.target_threshold))

        return aerial_image, printed_image

    def forward(self, params,num=None):

        if self.training:
            mask = self._pre_process(params)
        else:
            mask = params

        aerialNom, printedNom = self.corner_forward(mask, self.dose_nom, *self.focus_kernel)
        aerialMax, printedMax = self.corner_forward(mask, self.dose_max, *self.focus_kernel)
        aerialMin, printedMin = self.corner_forward(mask, self.dose_min, *self.defocus_kernel)

        pvband = torch.sum((printedMax >= self.target_threshold) != (printedMin >= self.target_threshold))

        if self.corners_forward:
            func = lambda d, *k: self.corner_forward(mask, d, *k)[1]

            if self.mask:
                index = sorted(random.sample(range(len(self.corners_config)),
                                     int(len(self.corners_config) * self.mask_ratio)))
                selected_corners = [self.corners_config[i] for i in index]
                mask_index_savedir = os.path.join(self.cfg.OUTPUT_DIR, "mask_index")
                with open(os.path.join(mask_index_savedir, f"selected_{num}.txt"), "a") as f:
                    line = ",".join(map(str, index))+"\n"
                    f.write(line)
                printedNoms = [func(corner[1], *corner[0]) for corner in selected_corners]
            else:
                printedNoms = [[func(d, *k) for d in self.dose]
                                    for k in [self.defocus_kernel, self.focus_kernel]]
                printedNoms = list(chain(*printedNoms))

        if self.training:
            if self.corners_forward:
                return mask, printedNoms, printedMax, printedMin, pvband
            return mask, [printedNom], printedMax, printedMin, pvband

        if self.corners_list:
            return mask, printedNoms[self.corners_list[0]], printedMax, printedMin, pvband

        return mask, printedNom, printedMax, printedMin, pvband
