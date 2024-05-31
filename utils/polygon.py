import os
import sys
import math
import copy
import time
import pickle
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def poly2img(polygons, sizeX, sizeY, scale=1.0): 
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    img = np.zeros([sizeY, sizeX], dtype=np.float32)
    for idx in range(len(polygons)): 
        polygon = np.array(polygons[idx])
        polygon = np.round(polygon * scale).astype(np.int64)
        img = cv2.fillPoly(img, [polygon], color=255)
    return img


def polysMin(polygons): 
    minX = None
    minY = None
    for idx in range(len(polygons)): 
        min1 = min(map(lambda x: x[0], polygons[idx]))
        min2 = min(map(lambda x: x[1], polygons[idx]))
        minX = min1 if minX is None else min(minX, min1)
        minY = min2 if minY is None else min(minY, min2)
    return (minX, minY)
def poly2imgShifted(polygons, sizeX, sizeY, scale=1.0, shifted=None): 
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    img = np.zeros([sizeY, sizeX], dtype=np.float32)
    minX, minY = polysMin(polygons) if shifted is None else shifted
    minval = np.array([[minX, minY, ], ])
    for idx in range(len(polygons)): 
        polygon = np.array(polygons[idx]) - minval
        polygon = np.round(polygon * scale).astype(np.int64)
        img = cv2.fillPoly(img, [polygon], color=255)
    return img[::-1, :]


def lines(polygon): 
    results = []
    for idx in range(len(polygon)): 
        results.append((polygon[idx], polygon[(idx+1)%len(polygon)]))
    return results
def dissect(polygon, lenCorner, lenUniform): 
    results = []
    for line in lines(polygon): 
        segments = []

        assert line[0][0] == line[1][0] or line[0][1] == line[1][1]
        if line[0][0] == line[1][0]: 
            axis = 1
        elif line[0][1] == line[1][1]: 
            axis = 0
        if line[0][axis] > line[1][axis]: 
            smaller = 1
            bigger = 0
        else: 
            smaller = 0
            bigger = 1
        length = line[bigger][axis] - line[smaller][axis]
        assert length > 0
        if length < 2 * lenCorner: 
            segments.append(line)
        else: 
            start = line[smaller][axis] + lenCorner
            end = line[bigger][axis] - lenCorner
            point1 = list(line[smaller])
            point1[axis] = start
            point1 = tuple(point1)
            point2 = list(line[bigger])
            point2[axis] = end
            point2 = tuple(point2)
            
            last1 = point1
            middle = start + round((end - start) / 2)
            segments1 = []
            segments1.append((line[smaller], point1))
            for pos1 in range(start+lenUniform, middle, lenUniform): 
                point = list(point1)
                point[axis] = pos1
                point = tuple(point)
                segments1.append((last1, point))
                last1 = point
                
            last2 = point2
            segments2 = []
            segments2.append((point2, line[bigger]))
            for pos2 in range(end-lenUniform, middle, -lenUniform): 
                point = list(point2)
                point[axis] = pos2
                point = tuple(point)
                segments2.append((point, last2))
                last2 = point

            if last2[axis] - last1[axis] < lenUniform: 
                segment1 = segments1.pop()
                segment2 = segments2.pop()
                segments1.append((segment1[0], segment2[1]))
            else: 
                segments1.append((last1, last2))
            
            segments.extend(segments1)
            segments.extend(segments2[::-1])

            for segment in segments: 
                assert abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]) >= min(lenCorner, lenUniform)

        if line[0][axis] > line[1][axis]: 
            segments = list(map(lambda x: (x[1], x[0]), segments[::-1]))
        
        results.extend(segments)

        # print(f"Line {line} -> {segments}")

    return results

def segs2poly(segments): 
    legalized = copy.deepcopy(segments)
    for idx in range(len(legalized)): 
        lastH = legalized[idx-1][0][0] == legalized[idx-1][1][0]
        lastV = legalized[idx-1][0][1] == legalized[idx-1][1][1]
        assert (lastH or lastV)
        thisH = legalized[idx][0][0] == legalized[idx][1][0]
        thisV = legalized[idx][0][1] == legalized[idx][1][1]
        assert (thisH or thisV)
        if lastH and thisV: 
            legalized[idx-1][1] = (legalized[idx-1][1][0], legalized[idx][0][1])
            legalized[idx][0] = (legalized[idx-1][1][0], legalized[idx][0][1])
        elif lastV and thisH: 
            legalized[idx-1][1] = (legalized[idx][0][0], legalized[idx-1][1][1])
            legalized[idx][0] = (legalized[idx][0][0], legalized[idx-1][1][1])

    polygon = []
    for elems in legalized: 
        for elem in elems: 
            if len(polygon) == 0: 
                polygon.append(elem)
            elif elem[0] != polygon[-1][0] or elem[1] != polygon[-1][1]: 
                polygon.append(elem)
    return polygon[:-1]

import matplotlib.pyplot as plt
if __name__ == "__main__": 
    polygon = [(3090, 15550), (3090, 15755), (2910, 15755), (2910, 16650), (2980, 16650), (2980, 15825), (3160, 15825), (3160, 15550)]
    dissected = dissect(polygon, lenCorner=35, lenUniform=70)
    reconstr = segs2poly(dissected)
    print(dissected)
    print(reconstr)

    plt.subplot(1, 2, 1)
    plt.imshow(poly2img([polygon], 20480, 20480, 0.1))
    plt.subplot(1, 2, 2)
    plt.imshow(poly2img([reconstr], 20480, 20480, 0.1))
    plt.show()
    