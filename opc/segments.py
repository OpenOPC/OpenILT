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


def cropSegments(segments, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700): 
    polygons = []
    for dissected in segments: 
        reconstr = poly.segs2poly(dissected)
        polygons.append(reconstr)
    print(f"In total {len(polygons)} shapes")

    reconstr = layout.createLayout(polygons, layer=layer, dbu=dbu)
    crops, coords = layout.getCrops(reconstr, layer=layer, sizeX=sizeX, sizeY=sizeY, 
                                    strideX=strideX, strideY=strideY, maxnum=None, verbose=False)
    print(f"In total {len(crops)} crops")

    return crops, coords


STEPS = 8
DECAY = 4
STEPSIZE = 8
BATCHSIZE = 16
MAXDIST = 24
SCALE = 0.125
if __name__ == "__main__": 
    filename = "data/gcd/gcd_45nm.gds" if len(sys.argv) < 2 else sys.argv[1]
    crop = False

    print(f"MB-OPC for {filename}")

    segments = dissectLayout(filename, layer=11, lenCorner=16, lenUniform=32, crop=crop)
    linked, flattened = flattenSegments(segments)
    linkedEPE = copy.deepcopy(linked)
    refsEPE = copy.deepcopy(flattened)

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
    crops, coords = cropSegments(linked, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700)
    bignom, bigmax, bigmin, origin = bigsim.simulate(crops, coords, batchsize=BATCHSIZE)

    predecessors = [None for _ in range(len(refsEPE))]
    successors = [None for _ in range(len(refsEPE))]
    inners = [None for _ in range(len(refsEPE))]
    outers = [None for _ in range(len(refsEPE))]
    inOverlaps = [None for _ in range(len(refsEPE))]
    outOverlaps = [None for _ in range(len(refsEPE))]
    inDists = [None for _ in range(len(refsEPE))]
    outDists = [None for _ in range(len(refsEPE))]
    
    index = 0
    groups = []
    id2group = {}
    for idx in range(len(linkedEPE)): 
        groups.append([])
        for jdx in range(len(linkedEPE[idx])): 
            groups[-1].append(index)
            id2group[index] = len(groups) - 1
            index += 1
    for idx in range(len(linkedEPE)): 
        for jdx in range(len(linkedEPE[idx])): 
            currIdx = groups[idx][jdx]
            prevIdx = groups[idx][jdx-1]
            nextIdx = groups[idx][(jdx+1)%len(groups[idx])]
            predecessors[currIdx] = prevIdx
            successors[currIdx] = nextIdx
    
    valids, lefts, rights, ups, downs = bigsim.validate(refsEPE, origin)
    directs = ["" for _ in range(len(valids))]
    for idx in range(len(valids)): 
        if not valids[idx]: 
            continue
        if lefts[idx]: 
            directs[idx] = "left"
        elif rights[idx]: 
            directs[idx] = "right"
        elif ups[idx]: 
            directs[idx] = "up"
        elif downs[idx]: 
            directs[idx] = "down"

    segsV = []
    segsH = []
    for idx, segment in enumerate(refsEPE): 
        assert segment[0][0] != segment[1][0] or segment[0][1] != segment[1][1]
        if segment[0][0] == segment[1][0]: 
            segsV.append(idx)
        elif segment[0][1] == segment[1][1]: 
            segsH.append(idx)
    segsV = sorted(segsV, key=lambda x: (refsEPE[x][0][0], (refsEPE[x][0][1] + refsEPE[x][1][1])/2))
    segsH = sorted(segsH, key=lambda x: (refsEPE[x][0][1], (refsEPE[x][0][0] + refsEPE[x][1][0])/2))
    print(f"segsH = {len(segsH)}, segsV = {len(segsV)}")

    def overlap(begin1, end1, begin2, end2): 
        if end1 < begin1: 
            begin1, end1 = end1, begin1
        if end2 < begin2: 
            begin2, end2 = end2, begin2
        if begin1 > end2 or begin2 > end1: 
            return 0
        elif begin1 > begin2: 
            return abs(end2 - begin1)
        elif begin2 > begin1: 
            return abs(end1 - begin2)
        return True

    limit = 180
    count = 0
    for idx in tqdm(range(len(segsV))): # left and right
        index = segsV[idx]
        if not valids[index]: 
            continue
        coordI = (round((refsEPE[index][0][0] + refsEPE[index][1][0])/2), 
                 round((refsEPE[index][0][1] + refsEPE[index][1][1])/2))
        
        indexIn = None
        indexOut = None

        # sameDirs = []
        diffDirs = []
        overlaps = {}
        distances = {}
        for jdx in range(idx, 0, -1): 
            jndex = segsV[jdx]
            if not valids[jndex]: 
                continue
            coordJ = (round((refsEPE[jndex][0][0] + refsEPE[jndex][1][0])/2), 
                      round((refsEPE[jndex][0][1] + refsEPE[jndex][1][1])/2))
            distance = np.abs(coordI[0]-coordJ[0])
            if distance > limit: 
                break
            overlapped = overlap(refsEPE[index][0][1], refsEPE[index][1][1], refsEPE[jndex][0][1], refsEPE[jndex][1][1])
            if overlapped > 0: 
                # if directs[index] == directs[jndex]: 
                #     sameDirs.append(jndex)
                if directs[index] != directs[jndex]: 
                    diffDirs.append(jndex)
                overlaps[jndex] = overlapped
                distances[jndex] = distance
        if directs[index] == "left" and len(diffDirs) > 0: 
            indexOut = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        elif directs[index] == "right" and len(diffDirs) > 0: 
            indexIn = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        else: 
            assert len(diffDirs) == 0
            
        # sameDirs = []
        diffDirs = []
        for jdx in range(idx, len(segsV), +1): 
            jndex = segsV[jdx]
            if not valids[jndex]: 
                continue
            coordJ = (round((refsEPE[jndex][0][0] + refsEPE[jndex][1][0])/2), 
                      round((refsEPE[jndex][0][1] + refsEPE[jndex][1][1])/2))
            distance = np.abs(coordI[0]-coordJ[0])
            if distance > limit: 
                break
            overlapped = overlap(refsEPE[index][0][1], refsEPE[index][1][1], refsEPE[jndex][0][1], refsEPE[jndex][1][1])
            if overlapped > 0: 
                # if directs[index] == directs[jndex]: 
                #     sameDirs.append(jndex)
                if directs[index] != directs[jndex]: 
                    diffDirs.append(jndex)
                overlaps[jndex] = overlapped
                distances[jndex] = distance
        if directs[index] == "left" and len(diffDirs) > 0: 
            indexIn = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        elif directs[index] == "right" and len(diffDirs) > 0: 
            indexOut = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        else: 
            assert len(diffDirs) == 0

        inners[index] = indexIn
        outers[index] = indexOut
        if not indexIn is None: 
            inOverlaps[index] = overlaps[indexIn]
            inDists[index] = distances[indexIn]
        if not indexOut is None: 
            outOverlaps[index] = overlaps[indexOut]
            outDists[index] = distances[indexOut]

        if not indexIn is None and not indexOut is None:  
        #     print(f"Segment {refsEPE[index]}-{directs[index]}, in-segment {refsEPE[indexIn] if not indexIn is None else None}-{directs[indexIn] if not indexIn is None else None}, out-segment {refsEPE[indexOut] if not indexOut is None else None}-{directs[indexOut] if not indexOut is None else None}")
            count += 1
        #     if count > 10: 
        #         break
            
    print(f" -> segsV Count = {count}/{len(segsV)}")

        
    count = 0
    for idx in tqdm(range(len(segsH))): 
        index = segsH[idx]
        if not valids[index]: 
            continue
        coordI = (round((refsEPE[index][0][0] + refsEPE[index][1][0])/2), 
                 round((refsEPE[index][0][1] + refsEPE[index][1][1])/2))
        
        indexIn = None
        indexOut = None

        # sameDirs = []
        diffDirs = []
        overlaps = {}
        distances = {}
        for jdx in range(idx, 0, -1): 
            jndex = segsH[jdx]
            if not valids[jndex]: 
                continue
            coordJ = (round((refsEPE[jndex][0][0] + refsEPE[jndex][1][0])/2), 
                      round((refsEPE[jndex][0][1] + refsEPE[jndex][1][1])/2))
            distance = np.abs(coordI[1]-coordJ[1])
            if distance > limit: 
                break
            overlapped = overlap(refsEPE[index][0][0], refsEPE[index][1][0], refsEPE[jndex][0][0], refsEPE[jndex][1][0])
            if overlapped > 0: 
                # if directs[index] == directs[jndex]: 
                #     sameDirs.append(jndex)
                if directs[index] != directs[jndex]: 
                    diffDirs.append(jndex)
                overlaps[jndex] = overlapped
                distances[jndex] = distance
        if directs[index] == "up" and len(diffDirs) > 0: 
            indexIn = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        elif directs[index] == "down" and len(diffDirs) > 0: 
            indexOut = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        else: 
            assert len(diffDirs) == 0
            
        # sameDirs = []
        diffDirs = []
        for jdx in range(idx, len(segsH), +1): 
            jndex = segsH[jdx]
            if not valids[jndex]: 
                continue
            coordJ = (round((refsEPE[jndex][0][0] + refsEPE[jndex][1][0])/2), 
                      round((refsEPE[jndex][0][1] + refsEPE[jndex][1][1])/2))
            distance = np.abs(coordI[1]-coordJ[1])
            if distance > limit: 
                break
            overlapped = overlap(refsEPE[index][0][0], refsEPE[index][1][0], refsEPE[jndex][0][0], refsEPE[jndex][1][0])
            if overlapped > 0: 
                # if directs[index] == directs[jndex]: 
                #     sameDirs.append(jndex)
                if directs[index] != directs[jndex]: 
                    diffDirs.append(jndex)
                overlaps[jndex] = overlapped
                distances[jndex] = distance
        if directs[index] == "up" and len(diffDirs) > 0: 
            indexOut = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        elif directs[index] == "down" and len(diffDirs) > 0: 
            indexIn = max(diffDirs, key=lambda x: (distances[x], overlaps[x]))
        else: 
            assert len(diffDirs) == 0

        inners[index] = indexIn
        outers[index] = indexOut
        if not indexIn is None: 
            inOverlaps[index] = overlaps[indexIn]
            inDists[index] = distances[indexIn]
        if not indexOut is None: 
            outOverlaps[index] = overlaps[indexOut]
            outDists[index] = distances[indexOut]

        if not indexIn is None and not indexOut is None:  
        #     print(f"Segment {refsEPE[index]}-{directs[index]}, in-segment {refsEPE[indexIn] if not indexIn is None else None}-{directs[indexIn] if not indexIn is None else None}, out-segment {refsEPE[indexOut] if not indexOut is None else None}-{directs[indexOut] if not indexOut is None else None}")
            count += 1
        #     if count > 10: 
        #         break
            
    print(f" -> segsH Count = {count}/{len(segsH)}")

    envs = []
    for idx in range(len(features)): 
        itself = features[idx]
        predecessor = features[predecessors[idx]] if not predecessors[idx] is None else tuple([0 for _ in range(len(itself))])
        successor = features[successors[idx]] if not successors[idx] is None else tuple([0 for _ in range(len(itself))])
        inner = features[inners[idx]] if not inners[idx] is None else tuple([0 for _ in range(len(itself))])
        outer = features[outers[idx]] if not outers[idx] is None else tuple([0 for _ in range(len(itself))])
        info = (inDists[inners[idx]] if not inners[idx] is None else 0, 
                inOverlaps[inners[idx]] if not inners[idx] is None else 0, 
                outDists[outers[idx]] if not outers[idx] is None else 0, 
                outOverlaps[outers[idx]] if not outers[idx] is None else 0, )
        env = itself + predecessor + successor + inner + outer + info
        envs.append(env)

    # MB-OPC
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
        crops, coords = cropSegments(linked, layer=11, dbu=1e-3, sizeX=1200, sizeY=1600, strideX=570, strideY=700)
        timePart = time.time() - timePart
        timePartAll += timePart
        
        timeSim = time.time()
        bignom, bigmax, bigmin, modified = bigsim.simulate(crops, coords, batchsize=BATCHSIZE)
        timeSim = time.time() - timeSim
        timeSimAll += timeSim

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
            if hmoves[idx] != 0: 
                node0 = (flattened[idx][0][0] + round(STEPSIZE * hmoves[idx]), flattened[idx][0][1])
                node1 = (flattened[idx][1][0] + round(STEPSIZE * hmoves[idx]), flattened[idx][1][1])
                if abs(node0[0] - reference[idx][0][0]) <= MAXDIST: 
                    flattened[idx][0] = node0
                if abs(node1[0] - reference[idx][1][0]) <= MAXDIST: 
                    flattened[idx][1] = node1
            if vmoves[idx] != 0: 
                node0 = (flattened[idx][0][0], flattened[idx][0][1] + round(STEPSIZE * vmoves[idx]))
                node1 = (flattened[idx][1][0], flattened[idx][1][1] + round(STEPSIZE * vmoves[idx]))
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
            STEPSIZE /= 2

    epe, viosAll, hmoves, vmoves = bigsim.checkEPE(reference, bignomBest, origin, distance=16, details=True)
    print(f"Final -> Fine-grained EPE violations: {epe}, {len(hmoves)}, {len(vmoves)}, {np.sum(hmoves!=0)+np.sum(vmoves!=0)}")
    epe, viosAll, hmoves, vmoves = bigsim.checkEPE(refsEPE, bignomBest, origin, distance=16, details=True)
    print(f"Final -> Coarse-grained EPE violations: {epe}")
    l2 = F.mse_loss(bignomBest, origin, reduction="sum").item()
    pvb = F.mse_loss(bigmaxBest, bigminBest, reduction="sum").item()
    print(f"Final -> L2: {l2:.3f}; PVB: {pvb:.3f}")
    print(f"Runtime -> Segmentation: {timeSegAll:.3f} s; Partition: {timePartAll:.3f} s; Simulation+Combination: {timeSimAll:.3f} s; Moving: {timeMoveAll:.3f} s")

    labels = []
    for idx, segment in enumerate(refsEPE): 
        labels.append(1 if hmoves[idx] != 0 or vmoves[idx] != 0 else 0)

    selectedE = []
    selectedL = []
    for idx in range(len(envs)): 
        if valids[idx]: 
            selectedE.append(envs[idx])
            selectedL.append(labels[idx])

    basename = os.path.basename(filename)[:-4]
    with open(f"newspots/{basename}.pkl", "wb") as fout: 
        pickle.dump((selectedE, selectedL), fout)