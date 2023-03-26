import sys
sys.path.append(".")
import math
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
# import pylitho.simple as lithosim
import pylitho.exact as lithosim

class Basic: 
    def __init__(self, litho=lithosim.LithoSim("./config/lithosimple.txt"), thresh=0.5, device=DEVICE): 
        self._litho  = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target, scale=1): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad(): 
            mask[mask >= self._thresh]  = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1: 
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return l2loss.item(), pvband.item()

    def sim(self, mask, target, scale=1): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad(): 
            mask[mask >= self._thresh]  = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1: 
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return mask, binaryNom



EPE_CONSTRAINT = 15
EPE_CHECK_INTERVEL = 40
MIN_EPE_CHECK_LENGTH = 80
EPE_CHECK_START_INTERVEL = 40

def boundaries(target, dtype=REALTYPE, device=DEVICE):
    boundary   = torch.zeros_like(target)
    corner     = torch.zeros_like(target)
    vertical   = torch.zeros_like(target)
    horizontal = torch.zeros_like(target)

    padded = func.pad(target[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper  = padded[2:,   1:-1]  == 1
    lower  = padded[:-2,  1:-1]  == 1
    left   = padded[1:-1, :-2]   == 1
    right  = padded[1:-1, 2:]    == 1
    upperleft  = padded[2:,  :-2] == 1
    upperright = padded[2:,  2:]  == 1
    lowerleft  = padded[:-2, :-2] == 1
    lowerright = padded[:-2, 2:]  == 1
    boundary = (target == 1)
    boundary[upper & lower & left & right & upperleft & upperright & lowerleft & lowerright] = False
    
    padded = func.pad(boundary[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper  = padded[2:,   1:-1]  == 1
    lower  = padded[:-2,  1:-1]  == 1
    left   = padded[1:-1, :-2]   == 1
    right  = padded[1:-1, 2:]    == 1
    center = padded[1:-1, 1:-1]  == 1

    vertical = center.clone()
    vertical[left & right] = False
    vsites = vertical.nonzero()
    vindices = np.lexsort((vsites[:, 0].detach().cpu().numpy(), vsites[:, 1].detach().cpu().numpy()))
    vsites = vsites[vindices]
    vstart = torch.cat((torch.tensor([True], device=vsites.device), vsites[:, 0][1:] != vsites[:, 0][:-1] + 1))
    vend   = torch.cat((vsites[:, 0][1:] != vsites[:, 0][:-1] + 1, torch.tensor([True], device=vsites.device)))
    vstart = vsites[(vstart == True).nonzero()[:, 0], :]
    vend   = vsites[(vend   == True).nonzero()[:, 0], :]
    vposes = torch.stack((vstart, vend), axis=2)
    
    horizontal = center.clone()
    horizontal[upper & lower] = False
    hsites = horizontal.nonzero()
    hindices = np.lexsort((hsites[:, 1].detach().cpu().numpy(), hsites[:, 0].detach().cpu().numpy()))
    hsites = hsites[hindices]
    hstart = torch.cat((torch.tensor([True], device=hsites.device), hsites[:, 1][1:] != hsites[:, 1][:-1] + 1))
    hend   = torch.cat((hsites[:, 1][1:] != hsites[:, 1][:-1] + 1, torch.tensor([True], device=hsites.device)))
    hstart = hsites[(hstart == True).nonzero()[:, 0], :]
    hend   = hsites[(hend   == True).nonzero()[:, 0], :]
    hposes = torch.stack((hstart, hend), axis=2)

    return vposes.float(), hposes.float()


def check(image, sample, target, direction):
    if direction == 'v':
        if ((target[sample[0, 0].long(), sample[0, 1].long() + 1] == 1) and (target[sample[0, 0].long(), sample[0, 1].long() - 1] == 0)): #left ,x small
            inner = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif ((target[sample[0, 0].long(), sample[0, 1].long() + 1] == 0) and (target[sample[0, 0].long(), sample[0, 1].long() - 1] == 1)): #right, x large
            inner = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    if direction == 'h':
        if((target[sample[0, 0].long() + 1, sample[0, 1].long()] == 1) and (target[sample[0, 0].long() - 1, sample[0, 1].long()] == 0)): #up, y small
            inner = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 0) and (target[sample[0, 0].long() - 1, sample[0, 1].long()] == 1): #low, y large
            inner = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    return inner, outer


def epecheck(mask, target, vposes, hposes):
    '''
    input: binary image tensor: (b, c, x, y); vertical points pair vposes: (N_v,4,2); horizontal points pair: (N_h, 4, 2), target image (b, c, x, y)
    output the total number of epe violations
    '''
    inner = 0
    outer = 0
    epeMap = torch.zeros_like(target)
    vioMap = torch.zeros_like(target)

    for idx in range(vposes.shape[0]):
        center = (vposes[idx, :, 0] + vposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0) #(1, 2)
        if (vposes[idx, 0, 1] - vposes[idx, 0, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'v')
        else:
            sampleY = torch.cat((torch.arange(vposes[idx, 0, 0] + EPE_CHECK_START_INTERVEL, center[0, 0] + 1, step = EPE_CHECK_INTERVEL), 
                                 torch.arange(vposes[idx, 0, 1] - EPE_CHECK_START_INTERVEL, center[0, 0],     step = -EPE_CHECK_INTERVEL))).unique()
            sample = vposes[idx, :, 0].repeat(sampleY.shape[0], 1)
            sample[:, 0] = sampleY
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'v')
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1

    for idx in range(hposes.shape[0]):
        center = (hposes[idx, :, 0] + hposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)
        if (hposes[idx, 1, 1] - hposes[idx, 1, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'h')
        else: 
            sampleX = torch.cat((torch.arange(hposes[idx, 1, 0] + EPE_CHECK_START_INTERVEL, center[0, 1] + 1, step = EPE_CHECK_INTERVEL), 
                                 torch.arange(hposes[idx, 1, 1] - EPE_CHECK_START_INTERVEL, center[0, 1],     step = -EPE_CHECK_INTERVEL))).unique()
            sample = hposes[idx, :, 0].repeat(sampleX.shape[0], 1)
            sample[:, 1] = sampleX
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'h')
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1
    return inner, outer, vioMap


class EPEChecker: 
    def __init__(self, litho=lithosim.LithoSim("./config/lithosimple.txt"), thresh=0.5, device=DEVICE): 
        self._litho  = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target, scale=1): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad(): 
            mask[mask >= self._thresh]  = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1: 
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            vposes, hposes = boundaries(target)
            epeIn, epeOut, _ =  epecheck(binaryNom, target, vposes, hposes)
        return epeIn, epeOut


import cv2
from adabox import proc, tools
class ShotCounter: 
    def __init__(self, litho=lithosim.LithoSim("./config/lithosimple.txt"), thresh=0.5, device=DEVICE): 
        self._litho  = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target=None, scale=1, shape=(512, 512)): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        image = torch.nn.functional.interpolate(mask[None, None, :, :], size=shape, mode="nearest")[0, 0]
        image = image.detach().cpu().numpy().astype(np.uint8)
        comps, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        rectangles = []
        for label in range(1, comps): 
            pixels = []
            for idx in range(labels.shape[0]): 
                for jdx in range(labels.shape[1]): 
                    if labels[idx, jdx] == label: 
                        pixels.append([idx, jdx, 0])
            pixels = np.array(pixels)
            x_data = np.unique(np.sort(pixels[:, 0]))
            y_data = np.unique(np.sort(pixels[:, 1]))
            if x_data.shape[0] == 1 or y_data.shape[0] == 1: 
                rectangles.append(tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max()))
                continue
            (rects, sep) = proc.decompose(pixels, 4)
            rectangles.extend(rects)
        return len(rectangles)

if __name__ == "__main__": 
    targetfile = sys.argv[1]
    maskfile = sys.argv[2]
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    test = Basic(litho, 0.5)
    epeCheck = EPEChecker(litho, 0.5)
    shotCount = ShotCounter(litho, 0.5)

    mask = cv2.imread(maskfile)[:, :, 0] / 255
    mask = cv2.resize(mask, (2048, 2048))
    if targetfile[:-4] == ".png": 
        target = cv2.imread(targetfile)[:, :, 0] / 255
        target = cv2.resize(target, (2048, 2048))
    else: 
        ref = glp.Design(targetfile, down=1)
        ref.center(2048, 2048, 0, 0)
        target = ref.mat(2048, 2048, 0, 0)

    l2, pvb = test.run(mask, target, scale=1)
    epeIn, epeOut = epeCheck.run(mask, target, scale=1)
    epe = epeIn + epeOut
    shot = shotCount.run(mask, shape=(512, 512))

    print(f"[{maskfile}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")
