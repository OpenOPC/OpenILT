import sys
sys.path.append(".")
import math
import multiprocessing as mp

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pylitho.simple as lithosim
# import pylitho.exact as lithosim

class Initializer: 
    def __init__(self): 
        pass
    
    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        pass


class PlainInit(Initializer): 
    def __init__(self): 
        super(PlainInit, self).__init__()
    
    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        if isinstance(design, glp.Design): 
            design = design.mat(sizeX, sizeY, offsetX, offsetY)
        target = torch.tensor(design, dtype=dtype, device=device)
        params = target.clone()
        return target, params


class PixelInit(Initializer): 
    def __init__(self): 
        super(PixelInit, self).__init__()
    
    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        if isinstance(design, glp.Design): 
            design = design.mat(sizeX, sizeY, offsetX, offsetY)
        target = torch.tensor(design, dtype=dtype, device=device)
        params = target * 2.0 - 1.0
        return target, params



def _distMatPolygon(polygon, canvas, offsets): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = np.ones([sizeX, sizeY]) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = np.tile(xs, [1, sizeY])
    ys = np.tile(ys, [sizeX, 1])
    
    frPt = polygon[-1]
    for toPt in polygon: 
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX: 
            frX, toX = toX, frX
        if frY > toY: 
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]
        
        dist1 = np.sqrt((frX - xs)**2 + (frY - ys)**2)
        dist2 = np.sqrt((toX - xs)**2 + (toY - ys)**2)
        
        dist = np.minimum(dist, np.minimum(dist1, dist2))

        if frX == toX: 
            mask = (frY <= ys) * (ys <= toY)
            new = np.minimum(dist, np.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY: 
            mask = (frX <= xs) * (xs <= toX)
            new = np.minimum(dist, np.abs(frY - ys))
            dist[mask] = new[mask]
            
        frPt = toPt
    return dist.T


def _distMatLegacy(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    
    mask = design.mat(sizeX=maxX-minX, sizeY=maxY-minY, offsetX=offsets[0], offsetY=offsets[1])
    dist = np.ones([maxX-minX, maxY-minY]) * ((maxX-minX) * (maxY-minY))
    for polygon in design.polygons: 
        tmp = _distMatPolygon(polygon, canvas, offsets)
        dist = np.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist



def _distMatPolygonTorch(polygon, canvas, offsets): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = torch.ones([sizeX, sizeY], dtype=REALTYPE, device=DEVICE) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = torch.tensor(np.tile(xs, [1, sizeY]), dtype=REALTYPE, device=DEVICE)
    ys = torch.tensor(np.tile(ys, [sizeX, 1]), dtype=REALTYPE, device=DEVICE)
    
    frPt = polygon[-1]
    for toPt in polygon: 
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX: 
            frX, toX = toX, frX
        if frY > toY: 
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]
        
        dist1 = torch.sqrt((frX - xs)**2 + (frY - ys)**2)
        dist2 = torch.sqrt((toX - xs)**2 + (toY - ys)**2)
        
        dist = torch.minimum(dist, torch.minimum(dist1, dist2))

        if frX == toX: 
            mask = (frY <= ys) * (ys <= toY)
            new = torch.minimum(dist, torch.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY: 
            mask = (frX <= xs) * (xs <= toX)
            new = torch.minimum(dist, torch.abs(frY - ys))
            dist[mask] = new[mask]
            
        frPt = toPt
    return dist.T


def _distMatTorch(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512], mask=None): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    
    if mask is None: 
        mask = design.mat(sizeX=maxX-minX, sizeY=maxY-minY, offsetX=offsets[0], offsetY=offsets[1])
    dist = torch.ones([maxX-minX, maxY-minY], dtype=REALTYPE, device=DEVICE) * ((maxX-minX) * (maxY-minY))
    for polygon in design.polygons: 
        tmp = _distMatPolygonTorch(polygon, canvas, offsets)
        dist = torch.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist


def _distMat(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    
    pool = mp.Pool(processes=mp.cpu_count()//2)
    procs = []
    for polygon in design.polygons: 
        proc = pool.apply_async(_distMatPolygon, (polygon, canvas, offsets))
        procs.append(proc)
    pool.close()
    pool.join()

    dist = np.ones([maxX-minX, maxY-minY]) * ((maxX-minX) * (maxY-minY))
    for proc in procs: 
        tmp = proc.get()
        dist = np.minimum(dist, tmp)
    mask = design.mat(sizeX=maxX-minX, sizeY=maxY-minY, offsetX=offsets[0], offsetY=offsets[1])
    dist[mask > 0] *= -1
    
    return dist

class LevelSetInit(Initializer): 
    def __init__(self): 
        super(LevelSetInit, self).__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        target = torch.tensor(design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device)
        params = torch.tensor(_distMat(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY]), dtype=REALTYPE, device=DEVICE, requires_grad=True)
        return target, params

class LevelSetInitTorch(Initializer): 
    def __init__(self): 
        super(LevelSetInitTorch, self).__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        target = torch.tensor(design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device)
        params = _distMatTorch(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY]).detach().clone().requires_grad_(True)
        return target, params

class LevelSetImageInitTrash(Initializer): 
    def __init__(self, kernelSize=3, iterations=1024): 
        super(LevelSetImageInit, self).__init__()
        self._kernelSize = kernelSize
        self._iterations = iterations

    def run(self, design, sizeX=None, sizeY=None, offsetX=None, offsetY=None, dtype=REALTYPE, device=DEVICE): 
        if isinstance(design, glp.Design): 
            design = design.mat(sizeX, sizeY, offsetX, offsetY)
        
        target = torch.tensor(design, dtype=dtype, device=device) if not isinstance(design, torch.Tensor) else design.to(dtype).to(device)
        kernel = torch.ones([1, 1, self._kernelSize, self._kernelSize], dtype=dtype, device=device)
        # Divisor
        divisor = func.conv2d(torch.ones_like(target)[None, None, :, :], kernel, padding='same')
        # In-shape values
        inshape = target.clone()[None, None, :, :]
        mask = inshape.clone()
        for idx in range(self._iterations): 
            conved = func.conv2d(mask, kernel, padding='same')
            mask = torch.zeros_like(mask)
            mask[torch.abs(conved - divisor) < 1e-3] = 1.0
            inshape += mask
            # print(f"In-shape: {torch.sum(mask).item()}, max: {torch.max(conved)}")
            if torch.sum(mask) == 0: 
                break
        inshape = -inshape
        # Out-shape values
        outshape = 1.0 - target.clone()[None, None, :, :]
        mask = outshape.clone()
        for idx in range(self._iterations): 
            conved = func.conv2d(mask, kernel, padding='same')
            mask = torch.zeros_like(mask)
            mask[torch.abs(conved - divisor) < 1e-3] = 1.0
            outshape += mask
            # print(f"Out-shape: {torch.sum(mask).item()}, max: {torch.max(conved)}")
            if torch.sum(mask) == 0: 
                break
        
        params = (inshape[0, 0] + outshape[0, 0]).detach().requires_grad_(True)
        return target, params

class LevelSetImageInit(Initializer): 
    def __init__(self): 
        super(LevelSetImageInit, self).__init__()
        self._initializer = LevelSetInitTorch()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        if isinstance(design, glp.Design): 
            design = design.mat(sizeX, sizeY, offsetX, offsetY)
        if isinstance(design, torch.Tensor): 
            design = design.detach().cpu().numpy()
        target = design
        
        polygons = []
        countours, hierarchy = cv2.findContours(design.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for countour in countours: 
            countour = countour.reshape([-1, 2])
            polygon = []
            for idx in range(countour.shape[0]): 
                coord = list(countour[idx])
                polygon.append(coord)
            polygons.append(polygon)
        
        design = glp.Design()
        design._polygons = polygons

        params = _distMatTorch(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY], mask=target).detach().clone().requires_grad_(True)
        target = torch.tensor(target, dtype=dtype, device=device)
        return target, params
        # return self._initializer.run(design, sizeX, sizeY, offsetX, offsetY, dtype=dtype, device=device)


if __name__ == "__main__": 
    ref = glp.Design(f"./benchmark/ICCAD2013/M1_test1.glp", down=1).mat(2048, 2048, 512, 512)
    target, params = LevelSetImageInit().run(ref, 2048, 2048, 0, 0)

    import levelset as ilt
    cfg   = ilt.LevelSetCfg("./config/pylevelset.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = ilt.LevelSetILT(cfg, litho)
    l2, pvb, bestMask = solver.solve(target, params)

    print(l2, pvb)
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.subplot(1, 2, 2)
    plt.imshow(bestMask)
    plt.show()