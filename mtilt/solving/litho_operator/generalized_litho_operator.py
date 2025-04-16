import logging
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtilt.config import configurable, CfgNode
from mtilt.utils.registry import Registry
from .build import META_LITHO_REGISTRY
from mtilt.config import get_cfg
from ..objectives import build_litho_loss, \
    build_pvb_l2_loss, build_pvb_loss, build_extra_loss

KERNEL_REGISTRY = Registry("KERNEL")
KERNEL_REGISTRY.__doc__ = """

""" # todo

logger = logging.getLogger(__name__)


def gradImage(image):
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = F.pad(image, (1, 1, 1, 1), mode='replicate')[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)


class _Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params):
        ctx.save_for_backward(params)
        mask = torch.zeros_like(params)
        mask[params < 0] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        params, = ctx.saved_tensors
        gradX, gradY = gradImage(params)
        l2norm = torch.sqrt(gradX ** 2 + gradY ** 2)
        return -l2norm * grad_output


class Binarize(nn.Module):
    def __init__(self):
        super(Binarize, self).__init__()
        pass

    def forward(self, params):
        return _Binarize.apply(params)

@KERNEL_REGISTRY.register()
class Simple_Kernel(object):

    def __init__(
            self,
            basedir: str="./kernel",
            defocus: bool=False,
            conjuncture: bool=False,
            combo: bool=False,
            device: str='cuda'
    ):
        self._basedir = basedir
        self._defocus = defocus
        self._conjuncture = conjuncture
        self._combo = combo
        self._device = device

        self._kernels = torch.load(self._kernel_file(), map_location=device).permute(2, 0, 1)
        self._scales = torch.load(self._scale_file(), map_location=device)

        self._knx, self._kny = self._kernels.shape[:2]

    @property
    def kernels(self):
        return self._kernels

    @property
    def scales(self):
        return self._scales

    def _kernel_file(self):
        filename = ""
        if self._defocus:
            filename = "defocus" + filename
        else:
            filename = "focus" + filename
        if self._conjuncture:
            filename = "ct_" + filename
        if self._combo:
            filename = "combo_" + filename
        filename = self._basedir + "/kernels/" + filename + ".pt"
        return filename

    def _scale_file(self):
        filename = self._basedir + "/scales/"
        if self._combo:
            return filename + "combo.pt"
        else:
            if self._defocus:
                return filename + "defocus.pt"
            else:
                return filename + "focus.pt"


def build_kernel(cfg):

    name = cfg.LITHO_OPERATOR.KERNEL.NAME
    kernel_list = cfg.LITHO_OPERATOR.KERNEL.TYPES
    kernel_params = cfg.LITHO_OPERATOR.KERNEL.PARAMS
    device = cfg.LITHO_OPERATOR.DEVICE
    dir = cfg.LITHO_OPERATOR.KERNEL.DIRNAME
    get_param = lambda x: {"defocus": x[0], "conjuncture":x[1], "combo": x[2]}
    return {kernel_name: KERNEL_REGISTRY.get(name)(device=device, basedir=dir,
        **get_param(eval(kernel_params).get(kernel_name, "None"))) for kernel_name in kernel_list}

def _maskFloat(mask, dose):
    return (dose * mask).to(eval(get_cfg().COMPLEXTYPE))

def _kernelMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    output = None
    if kernel.device != maskFFT.device:
        kernel = kernel.to(maskFFT.device)
    if maskFFT.shape[0] == 1:
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :knxh+1, :knyh+1] = maskFFT[:, :knxh+1, :knyh+1] * kernel[:kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :knxh+1, -knyh:] = maskFFT[:, :knxh+1, -knyh:] * kernel[:kernelNum, -(knxh+1):, :knyh]
        output[:, -knxh:, :knyh+1] = maskFFT[:, -knxh:, :knyh+1] * kernel[:kernelNum, :knxh, -(knyh+1):]
        output[:, -knxh:, -knyh:] = maskFFT[:, -knxh:, -knyh:] * kernel[:kernelNum, :knxh, :knyh]
    else:
        maskFFT = torch.unsqueeze(maskFFT, 1)
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :knxh+1, :knyh+1] = maskFFT[:, :, :knxh+1, :knyh+1] * kernel[None, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :knxh+1, -knyh:] = maskFFT[:, :, :knxh+1, -knyh:] * kernel[None, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, -knxh:, :knyh+1] = maskFFT[:, :, -knxh:, :knyh+1] * kernel[None, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, -knxh:, -knyh:] = maskFFT[:, :, -knxh:, -knyh:] * kernel[None, :kernelNum, :knxh, :knyh]
    return output

def _computeImage(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device:
        scale = scale.to(cmask.device)
    if len(cmask.shape) == 2:
        cmask = torch.unsqueeze(cmask, 0)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    if len(tmp.shape) == 3:
        if kernelNum == 1:
            return tmp[0]
        scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
        return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
    assert len(tmp.shape) == 4
    if kernelNum == 1:
        return tmp[:, 0]
    scale = scale[None, :kernelNum, None, None]
    return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)

def _convMatrix(cmask, dose, kernel, scale, kernelNum):
    image = _computeImage(cmask, kernel, scale, kernelNum)
    return image
def _convMask(mask, dose, kernel, scale, kernelNum):
    cmask = _maskFloat(mask, dose)
    image = _computeImage(cmask, kernel, scale, kernelNum)
    return image

class _LithoSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad):
        ctx.saved = (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad)
        return _convMask(mask, dose, kernel, scale, kernelNum)
    @staticmethod
    def backward(ctx, grad):
        (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad) = ctx.saved
        cpx0 = torch.mul(_convMask(mask, dose, kernelGradCT, scaleGradCT, kernelNumGradCT), grad)
        cpx1 = _convMatrix(cpx0, dose, kernelGrad, scaleGrad, kernelNumGrad)
        cpx2 = torch.mul(_convMask(mask, dose, kernelGrad, scaleGrad, kernelNumGrad), grad)
        cpx3 = _convMatrix(cpx2, dose, kernelGradCT, scaleGradCT, kernelNumGradCT)
        cpx4 = cpx1 + cpx3
        return cpx4.real, None, None, None, None, None, None, None, None, None, None


@META_LITHO_REGISTRY.register()
class BasicLithoSim(nn.Module):
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
            dose_nom: float,
            dose_min: float,
            dose_max: float,
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
            corners_forward: bool = False,
            device: str,
            dtype,
            # int_fields: list,
            # float_fields: list,
            multi_task: bool = False,
            initializer_type: str="pixel",
    ):
        super().__init__()
        self.cfg = config

        self.filter = filter
        self.kernels = kernels
        self.kernel_num = kernel_num

        # # defocus kernel
        self.focus_kernel = (self.kernels["focus"].kernels,
                                    self.kernels["focus"].scales,
                                    self.kernel_num,
                                    self.kernels["combo ct focus"].kernels,
                                    self.kernels["combo ct focus"].scales, 1,
                                    self.kernels["combo focus"].kernels,
                                    self.kernels["combo focus"].scales, 1)
        self.defocus_kernel = (self.kernels["defocus"].kernels,
                                   self.kernels["defocus"].scales,
                                   self.kernel_num,
                                   self.kernels["combo ct defocus"].kernels,
                                   self.kernels["combo ct defocus"].scales, 1,
                                   self.kernels["combo defocus"].kernels,
                                   self.kernels["combo defocus"].scales, 1)


        # dose
        self.dose_min = dose_min
        self.dose_max = dose_max
        self.dose_nom = dose_nom

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
                "dose_nom": cfg.LITHO_OPERATOR.DOSE_NOM,
                "dose_max": cfg.LITHO_OPERATOR.DOSE_MAX,
                "dose_min": cfg.LITHO_OPERATOR.DOSE_MIN,
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
                # "int_fields": cfg.LITHO_OPERATOR.INT_FIELDS,
                # "float_fields": cfg.LITHO_OPERATOR.FLOAT_FIELDS
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

    def forward(self, params):

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

            printedNoms = [[func(d, *k) for d in [self.dose_min, self.dose_nom, self.dose_max]]
                                for k in [self.defocus_kernel, self.focus_kernel]]
            printedNoms = list(chain(*printedNoms))

        if self.training:
            if self.corners_forward:
                return mask, printedNoms, printedMax, printedMin, pvband
            return mask, [printedNom], printedMax, printedMin, pvband

        return mask, printedNom, printedMax, printedMin, pvband

