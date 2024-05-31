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


def getCell(infile, layer, cell): 
    ly = infile
    cropped = ly.cell(cell)
    cropped.flatten(-1)
    layout = pya.Layout()
    layout.dbu = ly.dbu
    top = layout.create_cell(cell)
    layerFr = infile.layer(layer, 0)
    layerTo = layout.layer(layer, 0)
    for instance in cropped.each_shape(layerFr): 
        top.shapes(layerTo).insert(instance)
    return layout


def readLayout(filename, layer, crop=True): 
    infile = pya.Layout()
    infile.read(filename)
    bbox = infile.top_cell().bbox()
    left = bbox.left
    bottom = bbox.bottom
    right = bbox.right
    top = bbox.top
    print(f"Read layout of geometry ({left, bottom})-({right, top})")
    if crop: 
        cropped = cropLayout(infile, layer, left, bottom, right, top)
    else: 
        cropped = infile
    return cropped


def createLayout(polygons, layer, dbu): 
    layout = pya.Layout()
    layout.dbu = dbu
    top = layout.create_cell("TOP")
    layer = layout.layer(layer, 0)
    for polygon in polygons: 
        points = [pya.Point(point[0], point[1]) for point in polygon]
        instance = pya.SimplePolygon(points)
        top.shapes(layer).insert(instance)
    return layout


def cropLayout(infile, layer, beginX, beginY, endX, endY): 
    ly = infile
    scale = 0.001 / ly.dbu
    beginX = round(beginX * scale)
    beginY = round(beginY * scale)
    endX = round(endX * scale)
    endY = round(endY * scale)
    cbox = pya.Box.new(beginX, beginY, endX, endY)
    cropped = ly.clip(ly.top_cell().cell_index(), cbox)
    cropped = ly.cell(cropped)
    cropped.flatten(-1)
    layout = pya.Layout()
    layout.dbu = ly.dbu
    top = layout.create_cell("TOP")
    layerFr = infile.layer(layer, 0)
    layerTo = layout.layer(layer, 0)
    region = pya.Region(cropped.begin_shapes_rec(layerFr))
    region.merge()
    top.shapes(layerTo).insert(region)
    # for instance in cropped.each_shape(layerFr): 
    #     top.shapes(layerTo).insert(instance)
    return layout


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
            points = [(round((point[0]-minX)/scale), round((point[1]-minY)/scale)) for point in points]
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


def yieldShapes(infile, layer, fromX, fromY, toX, toY, anchor="min", verbose=True): # unit: nm
    ly = infile
    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell()
    topcell.flatten(-1)
    layer = ly.layer(layer, 0)
    if verbose: 
        print(f"Bounding box: {bbox}, selected layer: {layer}")
        
    scale = 0.001 / ly.dbu
    fromX = round(fromX * scale)
    fromY = round(fromY * scale)
    toX = round(toX * scale)
    toY = round(toY * scale)

    shapes = topcell.shapes(layer)
    for shape in shapes.each(): 
        polygons = []
        coords = []

        points = shape2points(shape, verbose)
        if len(points) > 0: 
            minX = min(map(lambda x: x[0], points))
            minY = min(map(lambda x: x[1], points))
            maxX = max(map(lambda x: x[0], points))
            maxY = max(map(lambda x: x[1], points))
            midX = (minX + maxX) / 2
            midY = (minY + maxY) / 2
            if anchor == "mid": 
                anchorX = midX
                anchorY = midY
            else: 
                anchorX = minX
                anchorY = minY
            reference = points
            countSkip = 0
            points = [(round((point[0]-anchorX)/scale), round((point[1]-anchorY)/scale)) for point in points]
            polygons.append(points)
            coords.append((round(minX/scale), round(minY/scale), round(maxX/scale), round(maxY/scale), round(midX/scale), round(midY/scale)))

            cbox = pya.Box.new(midX+fromX, midY+fromY, midX+toX, midY+toY)
            for neighbor in shapes.each_overlapping(cbox): 
                points = shape2points(neighbor, verbose)
                if len(points) > 0: 
                    if countSkip < 1 and len(points) == len(reference) and all(map(lambda x: points[x][0] == reference[x][0] and points[x][1] == reference[x][1], range(len(points)))): 
                        countSkip += 1
                        continue
                    minX = min(map(lambda x: x[0], points))
                    minY = min(map(lambda x: x[1], points))
                    maxX = max(map(lambda x: x[0], points))
                    maxY = max(map(lambda x: x[1], points))
                    midX = (minX + maxX) / 2
                    midY = (minY + maxY) / 2
                    points = [(round((point[0]-anchorX)/scale), round((point[1]-anchorY)/scale)) for point in points]
                    polygons.append(points)
                    coords.append((round(minX/scale), round(minY/scale), round(maxX/scale), round(maxY/scale), round(midX/scale), round(midY/scale)))
            assert countSkip == 1, f"ERROR: countSkip == {countSkip}"

        if len(polygons) > 0: 
            yield polygons, coords

def yieldCrops(infile, layer, sizeX, sizeY, strideX, strideY, offsetX=0, offsetY=0, verbose=True): # unit: nm
    ly = infile
    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell().cell_index()
    layer = ly.layer(layer, 0)
    left = bbox.left
    bottom = bbox.bottom
    right = bbox.right
    top = bbox.top
    if verbose: 
        print(f"Bounding box: {bbox}, selected layer: {layer}")
        
    scale = 0.001 / ly.dbu
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    strideX = round(strideX * scale)
    strideY = round(strideY * scale)
    offsetX = round(offsetX * scale)
    offsetY = round(offsetY * scale)
    left += offsetX
    bottom += offsetY

    countTotal = 0
    # for idx in range(left - (strideX-1), right + (strideX - 1), strideX): 
    for idx in range(left, right + (strideX - 1), strideX): 
        if verbose: 
            print(f"X position: {round(idx/scale)} -> {round((idx+sizeX)/scale)} / {round(right/scale)}, Y range: {round(bottom/scale)} - {round(top/scale)}, count={countTotal}")
        # for jdx in range(bottom - (strideY-1), top + (strideY-1), strideY): 
        for jdx in range(bottom, top + (strideY-1), strideY): 
            countTotal += 1
            cbox = pya.Box.new(idx, jdx, idx + sizeX, jdx + sizeY)
            cropped = ly.clip(topcell, cbox)
            cell = ly.cell(cropped)
            cell.flatten(-1)
            cell.name = f"Cropped_{idx}-{jdx}_{idx+sizeX}-{jdx+sizeY}"

            shapes = cell.shapes(layer)
            polygons = []
            for shape in shapes.each(): 
                points = shape2points(shape, verbose)

                points = [(round((point[0]-idx)/scale), round((point[1]-jdx)/scale)) for point in points]
                if len(points) > 0: 
                    polygons.append(points)
            
            if len(polygons) > 0: 
                yield polygons, (round(idx/scale), round(jdx/scale))


def getCrops(infile, layer, sizeX, sizeY, strideX, strideY, offsetX=0, offsetY=0, maxnum=None, verbose=True): # unit: nm
    iterator = yieldCrops(infile, layer, sizeX, sizeY, strideX, strideY, offsetX=offsetX, offsetY=offsetY, verbose=verbose)
    images = []
    coords = []
    for datum, (x, y) in iterator: 
        images.append(datum)
        coords.append((x, y))
        if not maxnum is None and len(images) >= maxnum: 
            break
    return images, coords


if __name__ == "__main__": 
    import polygon as poly

    infile = pya.Layout()
    infile.read("gds/gcd_45nm.gds")
    cropped = cropLayout(infile, 11, 0, 0, 32000, 32000)
    shapes, coords = getShapes(cropped, layer=11, maxnum=None, verbose=True)
    polygons = []
    for datum, coord in zip(shapes, coords): 
        polygon = list(map(lambda x: (x[0]+coord[0], x[1]+coord[1]), datum))
        dissected = poly.dissect(polygon, lenCorner=35, lenUniform=70)
        reconstr = poly.segs2poly(dissected)
        polygons.append(reconstr)
    print(f"In total {len(polygons)} shapes") 

    layout = createLayout(polygons, layer=11, dbu=1e-3)
    cropped.write("trivial/test0.gds")
    layout.write("trivial/test1.gds")

    shapes, coords = getCrops(layout, layer=11, sizeX=570, sizeY=1570, strideX=190, strideY=1400, maxnum=None, verbose=True)
    count = 0
    maxnum = 0
    for datum, coord in zip(shapes, coords): 
        count += 1
        if len(datum) > maxnum: 
            maxnum = len(datum)
        # print(f"Shapes: {coord}, {datum}")
    print(f"Count = {count}")
