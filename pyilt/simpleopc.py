import os
import sys
sys.path.append(".")
import math
import glob
import copy
import time
import random
import pickle

import numpy as np
import cv2
from sklearn.cluster import *
from sklearn.neighbors import *
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils.layout as layout
import utils.polygon as plg
import pylitho.simple as lithosim

import pycommon.glp as glp
import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cropSegments(segments, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700, offsetX=0, offsetY=0): 
    polygons = []
    for dissected in segments: 
        reconstr = plg.segs2poly(dissected)
        polygons.append(reconstr)
    print(f"In total {len(polygons)} shapes")

    reconstr = layout.createLayout(polygons, layer=layer, dbu=dbu)
    crops, coords = layout.getCrops(reconstr, layer=layer, sizeX=sizeX, sizeY=sizeY, 
                                    strideX=strideX, strideY=strideY, offsetX=offsetX, offsetY=offsetY, 
                                    maxnum=None, verbose=False)
    print(f"In total {len(crops)} crops")

    return crops, coords

def checkEPE(segments, bignom, origin, distance=16, scale=1, details=False): 
        bignom = bignom.detach().cpu().numpy()
        bignom = (bignom > 0.499)
        origin = origin.detach().cpu().numpy()
        origin = (origin > 0.499)

        begins = list(map(lambda x: x[0], segments))
        ends = list(map(lambda x: x[1], segments))
        begins = np.array(begins, dtype=np.int32)
        ends = np.array(ends, dtype=np.int32)
        mids = (begins + ends) / 2
        begins = np.round(scale*begins).astype(dtype=np.int32)
        ends = np.round(scale*ends).astype(dtype=np.int32)
        mids = np.round(scale*mids).astype(dtype=np.int32)
        equals = (ends == begins)

        allequal = np.logical_and(equals[:, 0], equals[:, 1])
        verticals = np.logical_and(equals[:, 0], np.logical_not(allequal))
        horizontal = np.logical_and(equals[:, 1], np.logical_not(allequal))
        lefts = mids.copy()
        lefts[:, 0] += 2
        lefts = origin[lefts[:, 1], lefts[:, 0]]
        lefts = np.logical_and(verticals, lefts)
        rights = mids.copy()
        rights[:, 0] -= 2
        rights = origin[rights[:, 1], rights[:, 0]]
        rights = np.logical_and(verticals, rights)
        ups = mids.copy()
        ups[:, 1] -= 2
        ups = origin[ups[:, 1], ups[:, 0]]
        ups = np.logical_and(horizontal, ups)
        downs = mids.copy()
        downs[:, 1] += 2
        downs = origin[downs[:, 1], downs[:, 0]]
        downs = np.logical_and(horizontal, downs)
        lr = np.logical_and(lefts, rights)
        ud = np.logical_and(ups, downs)
        lefts = np.logical_and(lefts, np.logical_not(lr))
        rights = np.logical_and(rights, np.logical_not(lr))
        ups = np.logical_and(ups, np.logical_not(ud))
        downs = np.logical_and(downs, np.logical_not(ud))

        inners = mids.copy()
        inners[lefts, 0] += round(scale * distance)
        inners[rights, 0] -= round(scale * distance)
        inners[ups, 1] -= round(scale * distance)
        inners[downs, 1] += round(scale * distance)
        outers = mids.copy()
        outers[lefts, 0] -= round(scale * distance)
        outers[rights, 0] += round(scale * distance)
        outers[ups, 1] += round(scale * distance)
        outers[downs, 1] -= round(scale * distance)

        validsIn = origin[inners[:, 1], inners[:, 0]]
        validsOut = np.logical_not(origin[outers[:, 1], outers[:, 0]])
        viosIn = np.logical_not(bignom[inners[:, 1], inners[:, 0]])
        viosIn = np.logical_and(viosIn, validsIn)
        viosOut = bignom[outers[:, 1], outers[:, 0]]
        viosOut = np.logical_and(viosOut, validsOut)
        viosAll = np.logical_or(viosIn, viosOut)

        epe = np.sum(viosAll)
        print(f"EPE violations: {epe}={np.sum(viosIn)}+{np.sum(viosOut)}/{np.sum(validsIn)+np.sum(validsOut)}/{2*len(segments)}")

        if details: 
            hmoves = np.zeros((len(segments), ), dtype=np.int32)
            hmoves[np.logical_and(lefts, viosIn)] = -1
            hmoves[np.logical_and(rights, viosIn)] = +1
            hmoves[np.logical_and(lefts, viosOut)] = +1
            hmoves[np.logical_and(rights, viosOut)] = -1
            vmoves = np.zeros((len(segments), ), dtype=np.int32)
            vmoves[np.logical_and(ups, viosIn)] = +1
            vmoves[np.logical_and(downs, viosIn)] = -1
            vmoves[np.logical_and(ups, viosOut)] = -1
            vmoves[np.logical_and(downs, viosOut)] = +1
            return epe, viosAll, hmoves, vmoves

        return epe

STEPS = 8
DECAY = 4
STEPSIZE = 8
BATCHSIZE = 16
MAXDIST = 24
SCALE = 1
if __name__ == "__main__": 
    size = (2048, 2048)
    shifted = (384+512, 384+512)

    lenCorner = 16
    lenUniform = 32

    for index in range(1, 11): 
        StepSize = STEPSIZE

        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{index}.glp", down=SCALE)
        design.center(size[0], size[1], 0, 0)
        polygons = design.polygons
        for polygon in polygons: 
            for point in polygon: 
                point[0] = int(point[0])
                point[1] = int(point[1])

        segments = []
        for polygon in polygons: 
            dissected = plg.dissect(polygon, lenCorner=lenCorner, lenUniform=lenUniform)
            segments.append(dissected)
        linked = []
        flattened = []
        for idx, elems in enumerate(segments): 
            elems = list(map(lambda x: list(x), elems))
            linked.append(elems)
            flattened.extend(elems)
        reference = copy.deepcopy(flattened)
        linkedEPE = copy.deepcopy(linked)
        refsEPE = copy.deepcopy(flattened)
        
        coords = list(map(lambda x: plg.segs2poly(x), linked))
        target = plg.poly2img(coords, size[0], size[1], scale=SCALE).copy() / 255
        target = torch.tensor(target, dtype=torch.float32, device=DEVICE)

        litho = lithosim.LithoSim("./config/lithosimple.txt")
        bignom, bigmax, bigmin = litho(target)
        l2, pvb, epe, shot = evaluation.evaluate(target, target, litho, scale=SCALE, shots=False)
        print(f"[Testcase {index} Initialized]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")

        epeBest = 1e9
        modifiedBest = None
        bignomBest = None
        bigmaxBest = None
        bigminBest = None
        timePartAll = 0
        timeMoveAll = 0
        timeSimAll = 0
        for step in range(STEPS): 
            print(f"Step {step}")

            # for jdx in range(len(crops)): 
                # print(coords[jdx], crops[jdx])
            
            timeSim = time.time()
            litho = lithosim.LithoSim("./config/lithosimple.txt")
            coords = list(map(lambda x: plg.segs2poly(x), linked))
            modified = plg.poly2img(coords, size[0], size[1], scale=SCALE).copy() / 255
            modified = torch.tensor(modified, dtype=torch.float32, device=DEVICE)
            bignom, bigmax, bigmin = litho(modified)
            l2, pvb, epe, shot = evaluation.evaluate(modified, target, litho, scale=SCALE, shots=False)
            timeSim = time.time() - timeSim
            timeSimAll += timeSim

            # polygons = []
            # for dissected in linked: 
            #     reconstr = plg.segs2poly(dissected)
            #     polygons.append(reconstr)
            # print(f"In total {len(polygons)} shapes")
            # reconstr = layout.createLayout(polygons, layer=11, dbu=1e-3)
            # reconstr.write("tmp.gds")

            timeMove = time.time()
            epe, viosAll, hmoves, vmoves = checkEPE(reference, bignom, target, distance=16, scale=SCALE, details=True)
            print(f" -> Fine-grained EPE violations: {epe}, {len(hmoves)}, {len(vmoves)}, {np.sum(hmoves!=0)+np.sum(vmoves!=0)}")
            l2, pvb, epe, shot = evaluation.evaluate(modified, target, litho, scale=SCALE, shots=False)
            print(f"[Testcase {index} Step {step}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")
            timeMove = time.time() - timeMove
            timeMoveAll += timeMove

            for idx, elems in enumerate(flattened): 
                assert hmoves[idx] == 0 or vmoves[idx] == 0
                if hmoves[idx] != 0: 
                    node0 = (flattened[idx][0][0] + round(StepSize * hmoves[idx]), flattened[idx][0][1])
                    node1 = (flattened[idx][1][0] + round(StepSize * hmoves[idx]), flattened[idx][1][1])
                    if abs(node0[0] - reference[idx][0][0]) <= MAXDIST: 
                        flattened[idx][0] = node0
                    if abs(node1[0] - reference[idx][1][0]) <= MAXDIST: 
                        flattened[idx][1] = node1
                if vmoves[idx] != 0: 
                    node0 = (flattened[idx][0][0], flattened[idx][0][1] + round(StepSize * vmoves[idx]))
                    node1 = (flattened[idx][1][0], flattened[idx][1][1] + round(StepSize * vmoves[idx]))
                    if abs(node0[1] - reference[idx][0][1]) <= MAXDIST: 
                        flattened[idx][0] = node0
                    if abs(node1[1] - reference[idx][1][1]) <= MAXDIST: 
                        flattened[idx][1] = node1

            if epeBest > epe: 
                epeBest = epe
                modifiedBest = modified
                bignomBest = bignom
                bigmaxBest = bigmax
                bigminBest = bigmin

            if step > 0 and step % DECAY == 0: 
                StepSize /= 2

        target = target.detach().cpu().numpy()
        modified = modified.detach().cpu().numpy()
        printed = bignom.detach().cpu().numpy()
        printed[printed >= 0.5] = 1
        printed[printed < 0.5] = 0
        cv2.imwrite(f"tmp/SimpleOPC_target{index}.png", cv2.resize(target * 255, (2048, 2048)))
        cv2.imwrite(f"tmp/SimpleOPC_mask{index}.png",  cv2.resize(modified * 255, (2048, 2048)))
        cv2.imwrite(f"tmp/SimpleOPC_resist{index}.png",  cv2.resize(printed * 255, (2048, 2048)))




