import os
import sys
sys.path.append(".")
import time
import copy
import pickle

import cv2 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils.polygon as poly
import utils.layout as layout
import iccad13 as litho


def dissectLayout(filename, layer=11, lenCorner=35, lenUniform=70, crop=True): 
    infile = layout.readLayout(filename, layer, crop)
    shapes, poses = layout.getShapes(infile, layer=layer, maxnum=None, verbose=False)
    segments = []
    for datum, coord in zip(shapes, poses): 
        polygon = list(map(lambda x: (x[0]+coord[0], x[1]+coord[1]), datum))
        dissected = poly.dissect(polygon, lenCorner=lenCorner, lenUniform=lenUniform)
        segments.append(dissected)
    print(f"In total {len(segments)} shapes") 

    return segments


def flattenSegments(segments): 
    linked = []
    results = []
    for idx, elems in enumerate(segments): 
        elems = list(map(lambda x: list(x), elems))
        linked.append(elems)
        results.extend(elems)

    return linked, results


def cropSegments(segments, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700, fromzero=True): 
    polygons = []
    for dissected in segments: 
        reconstr = poly.segs2poly(dissected)
        polygons.append(reconstr)
    print(f"In total {len(polygons)} shapes")

    reconstr = layout.createLayout(polygons, layer=layer, dbu=dbu)
    crops, coords = layout.getCrops(reconstr, layer=layer, sizeX=sizeX, sizeY=sizeY, strideX=strideX, strideY=strideY, 
                                    maxnum=None, fromzero=fromzero, verbose=False)
    print(f"In total {len(crops)} crops")

    return crops, coords


STEPS = 8
DECAY = 4
STEPSIZE = 8
BATCHSIZE = 64
MAXDIST = 24
SCALE = 0.125
if __name__ == "__main__": 
    filename = "benchmark/gcd_45nm.gds" if len(sys.argv) < 2 else sys.argv[1]
    crop = False

    print(f"MB-OPC for {filename}")

    segments = dissectLayout(filename, layer=11, lenCorner=16, lenUniform=32, crop=crop)
    linked, flattened = flattenSegments(segments)
    linkedEPE = copy.deepcopy(linked)
    refsEPE = copy.deepcopy(flattened)

    stepsize = STEPSIZE

    features = []
    for idx, segment in enumerate(refsEPE): 
        point1 = segment[0]
        point2 = segment[1]
        directH = 1 if point1[0] == point2[0] else 0 
        directV = 1 if point1[1] == point2[1] else 0 
        assert directH or directV and not (directH and directV)
        centerX = (point1[0] + point1[0]) / 2
        centerY = (point1[1] + point1[1]) / 2
        length = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        lastH = 1 if refsEPE[idx-1][0][0] == refsEPE[idx-1][1][0] else 0 
        lastV = 1 if refsEPE[idx-1][0][1] == refsEPE[idx-1][1][1] else 0
        assert lastH or lastV and not (lastH and lastV)
        turn = 1 if (lastH and directV) or (lastV and directH) else 0
        features.append((length, directH, directV, centerX, centerY, turn))

    timeSegAll = time.time()
    segments = dissectLayout(filename, layer=11, lenCorner=16, lenUniform=32, crop=crop)
    linked, flattened = flattenSegments(segments)
    reference = copy.deepcopy(flattened)
    timeSegAll = time.time() - timeSegAll

    lengths = []
    for segment in flattened: 
        lengths.append(abs(segment[0][0]-segment[1][0]) + abs(segment[0][1]-segment[1][1]))
    counts, bins = np.histogram(lengths)
    print("Histogram")
    print(" ->", counts)
    print(" ->", bins)

    bigsim = litho.PatchSim(litho.LithoSim(), sizeX=1200, sizeY=1600, scale=SCALE)
    crops, coords = cropSegments(linked, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700, fromzero=True)
    bignom, bigmax, bigmin, origin = bigsim.simulate(crops, coords, batchsize=BATCHSIZE)
    
    valids, lefts, rights, ups, downs = bigsim.validate(refsEPE, origin)

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
        timePart = time.time()
        crops, coords = cropSegments(linked, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700, fromzero=True)
        timePart = time.time() - timePart
        timePartAll += timePart
        
        timeSim = time.time()
        bignom, bigmax, bigmin, modified = bigsim.simulate(crops, coords, batchsize=BATCHSIZE)
        timeSim = time.time() - timeSim
        timeSimAll += timeSim

        # polygons = []
        # for dissected in linked: 
        #     reconstr = poly.segs2poly(dissected)
        #     polygons.append(reconstr)
        # print(f"In total {len(polygons)} shapes")
        # reconstr = layout.createLayout(polygons, layer=11, dbu=1e-3)
        # reconstr.write("tmp.gds")

        timeMove = time.time()
        epe, viosAll, hmoves, vmoves = bigsim.checkEPE(reference, bignom, origin, distance=16, details=True)
        print(f" -> Fine-grained EPE violations: {epe}, {len(hmoves)}, {len(vmoves)}, {np.sum(hmoves!=0)+np.sum(vmoves!=0)}")
        epe = bigsim.checkEPE(refsEPE, bignom, origin, distance=16, details=False)
        print(f" -> Coarse-grained EPE violations: {epe}")
        l2 = F.mse_loss(bignom, origin, reduction="sum").item()
        pvb = F.mse_loss(bigmax, bigmin, reduction="sum").item()
        print(f" -> L2: {l2:.3f}; PVB: {pvb:.3f}")
        timeMove = time.time() - timeMove
        timeMoveAll += timeMove

        for idx, elems in enumerate(flattened): 
            assert hmoves[idx] == 0 or vmoves[idx] == 0
            # if hmoves[idx] != 0: 
            #     flattened[idx][0] = (flattened[idx][0][0] + round(stepsize * hmoves[idx]), flattened[idx][0][1])
            #     flattened[idx][1] = (flattened[idx][1][0] + round(stepsize * hmoves[idx]), flattened[idx][1][1])
            # if vmoves[idx] != 0: 
            #     flattened[idx][0] = (flattened[idx][0][0], flattened[idx][0][1] + round(stepsize * vmoves[idx]))
            #     flattened[idx][1] = (flattened[idx][1][0], flattened[idx][1][1] + round(stepsize * vmoves[idx]))
            if hmoves[idx] != 0: 
                node0 = (flattened[idx][0][0] + round(stepsize * hmoves[idx]), flattened[idx][0][1])
                node1 = (flattened[idx][1][0] + round(stepsize * hmoves[idx]), flattened[idx][1][1])
                if abs(node0[0] - reference[idx][0][0]) <= MAXDIST: 
                    flattened[idx][0] = node0
                if abs(node1[0] - reference[idx][1][0]) <= MAXDIST: 
                    flattened[idx][1] = node1
            if vmoves[idx] != 0: 
                node0 = (flattened[idx][0][0], flattened[idx][0][1] + round(stepsize * vmoves[idx]))
                node1 = (flattened[idx][1][0], flattened[idx][1][1] + round(stepsize * vmoves[idx]))
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
            stepsize /= 2

    epe, viosAll, hmoves, vmoves = bigsim.checkEPE(reference, bignomBest, origin, distance=16, details=True)
    print(f"Final -> Fine-grained EPE violations: {epe}, {len(hmoves)}, {len(vmoves)}, {np.sum(hmoves!=0)+np.sum(vmoves!=0)}")
    epe, viosAll, hmoves, vmoves = bigsim.checkEPE(refsEPE, bignomBest, origin, distance=16, details=True)
    print(f"Final -> Coarse-grained EPE violations: {epe}")
    l2 = F.mse_loss(bignomBest, origin, reduction="sum").item()
    pvb = F.mse_loss(bigmaxBest, bigminBest, reduction="sum").item()
    print(f"Final -> L2: {l2:.3f}; PVB: {pvb:.3f}")
    print(f"Runtime -> Segmentation: {timeSegAll:.3f} s; Partition: {timePartAll:.3f} s; Simulation+Combination: {timeSimAll:.3f} s; Moving: {timeMoveAll:.3f} s")

    origin = (origin.detach().cpu().numpy() > 0.499).astype(np.float32)
    modified = (modifiedBest.detach().cpu().numpy() > 0.499).astype(np.float32)
    bignom = (bignomBest.detach().cpu().numpy() > 0.499).astype(np.float32)
    basename = os.path.basename(filename)[:-4]
    cv2.imwrite(f"tmp/{basename}_origin.png", np.flip(origin*255, axis=0))
    cv2.imwrite(f"tmp/{basename}_modified.png", np.flip(modified*255, axis=0))
    cv2.imwrite(f"tmp/{basename}_simed.png", np.flip(bignom*255, axis=0))


'''
Final -> Fine-grained EPE violations: 23542, 194937, 194937, 23489
EPE violations: 23542=7514+16034/383541/389874
Final -> Coarse-grained EPE violations: 23542
Final -> L2: 452295.906; PVB: 145118.938
Runtime -> Segmentation: 1.373 s; Partition: 29.795 s; Simulation+Combination: 50.358 s; Moving: 3.726 s
'''