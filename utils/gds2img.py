import os
import sys
import math
import pickle
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm

try: 
    import pya
except Exception: 
    import klayout.db as pya

def readLayout(filename, layer, crop=True): 
    infile = pya.Layout()
    infile.read(filename)
    bbox = infile.top_cell().bbox()
    left = bbox.left
    bottom = bbox.bottom
    right = bbox.right
    top = bbox.top
    print(f"Read layout of geometry ({left, bottom})-({right, top})")
    return infile


def shape2points(shape, verbose=False): 
    points = []
    if shape.is_box(): 
        box = shape.box
        points.append((box.left, box.bottom))
        points.append((box.left, box.top))
        points.append((box.right, box.top))
        points.append((box.right, box.bottom))
        if verbose == "debug": 
            print(f"Box: ({box.bottom}, {box.left}, {box.top}, {box.right})")
    elif shape.is_path(): 
        path = shape.path
        polygon = path.simple_polygon()
        for point in polygon.each_point(): 
            points.append((point.x, point.y))
        if verbose == "debug": 
            print(f"Path: ({polygon})")
            for point in polygon.each_point(): 
                print(f" -> Point: {point}")
    elif shape.is_polygon(): 
        polygon = shape.polygon
        polygon = polygon.to_simple_polygon()
        for point in polygon.each_point(): 
            points.append((point.x, point.y))
        if verbose == "debug": 
            print(f"Polygon: ({polygon})")
            for point in polygon.each_point(): 
                print(f" -> Point: {point.x}, {point.y}")
    valid = shape.is_box() or shape.is_path() or shape.is_polygon() or shape.is_text() or shape.is_edge()
    assert valid, f"ERROR: Invalid shape: {shape}"

    return points


def yieldShape(infile, layer, verbose=True): # unit: nm
    ly = infile
    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell()
    topcell.flatten(-1)
    layer = ly.layer(layer, 0)
    if verbose: 
        print(f"Bounding box: {bbox}, selected layer: {layer}")
        
    scale = 0.001 / ly.dbu

    shapes = topcell.shapes(layer)
    for shape in shapes.each(): 
        points = shape2points(shape, verbose)

        if len(points) > 0: 
            minX = min(map(lambda x: x[0], points))
            minY = min(map(lambda x: x[1], points))
            maxX = max(map(lambda x: x[0], points))
            maxY = max(map(lambda x: x[1], points))
            midX = (minX + maxX) / 2
            midY = (minY + maxY) / 2
            points = [(round(point[0]/scale), round(point[1]/scale)) for point in points]
            # points = [(round((point[0]-minX)/scale), round((point[1]-minY)/scale)) for point in points]
            yield points, (round(minX/scale), round(minY/scale), round(maxX/scale), round(maxY/scale), round(midX/scale), round(midY/scale))


def getShapes(infile, layer, maxnum=None, verbose=True): 
    iterator = yieldShape(infile, layer, verbose)
    polygons = []
    coords = []
    for datum, coord in iterator: 
        polygons.append(datum)
        coords.append(coord)
        if not maxnum is None and len(polygons) >= maxnum: 
            break
    minx = min(map(lambda x: x[0], coords))
    miny = min(map(lambda x: x[1], coords))
    coords = list(map(lambda x: (x[0]-minx, x[1]-miny, x[2]-minx, x[3]-miny, x[4]-minx, x[5]-miny), coords))
    for points in coords: 
        for elem in points: 
            assert elem >= 0, f"WRONG: {points}"
    
    return polygons, coords


def poly2img(polygons, sizeX, sizeY, scale=1.0): 
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    img = np.zeros([sizeY, sizeX], dtype=np.float32)
    mins = [1e12, 1e12]
    for idx in range(len(polygons)): 
        mins[0] = min(mins[0], np.min(np.array(polygons[idx])[:, 0]))
        mins[1] = min(mins[1], np.min(np.array(polygons[idx])[:, 1]))
    for idx in range(len(polygons)): 
        polygon = np.array(polygons[idx])
        polygon[:, 0] -= mins[0]
        polygon[:, 1] -= mins[1]
        polygon = np.round(polygon * scale).astype(np.int64)
        img = cv2.fillPoly(img, [polygon], color=255)
    return img


if __name__ == "__main__": 
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import matplotlib.pyplot as plt
    import pylitho.exact as lithosim
    import torch
    filename = "utils/cell68.gds"
    layer = 3
    infile = readLayout(filename, layer, crop=False)
    polygons, coords = getShapes(infile, layer, maxnum=None, verbose=True)
    image = poly2img(polygons, sizeX=1024, sizeY=1024, scale=1.0)/255
    image = np.pad(image, (512, 512))
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    mask = torch.tensor(image, dtype=torch.float32)
    printedNom, printedMax, printedMin = litho(mask)
    printed = np.round(printedNom.detach().cpu().numpy())
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(printed)
    plt.show()
    
