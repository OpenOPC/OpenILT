import sys
sys.path.append(".")
import math

import numpy as np 
import cv2

class Design: 
    def __init__(self, filename=None, down=1): 
        self._filename = filename
        self._polygons = []
        if filename is None: 
            return
        with open(filename, "r") as fin: 
            lines = fin.readlines()
        for line in lines: 
            splited = line.strip().split()
            if len(splited) < 7: 
                continue
            if splited[0] == "RECT": 
                info = splited[3:7]
                frX = int(info[0])
                frY = int(info[1])
                toX = frX + int(info[2])
                toY = frY + int(info[3])
                coords = [[frX//down, frY//down], [frX//down, toY//down], 
                          [toX//down, toY//down], [toX//down, frY//down]]
                self._polygons.append(coords)
            elif splited[0] == "PGON": 
                info = splited[3:]
                coords = []
                for idx in range(0, len(info), 2): 
                    coordX = int(info[idx])
                    coordY = int(info[idx+1])
                    coords.append([coordX//down, coordY//down])
                self._polygons.append(coords)
    
    @property
    def polygons(self): 
        return self._polygons

    def range(self): 
        minX = 1e12
        minY = 1e12
        maxX = -1e12
        maxY = -1e12
        for polygon in self._polygons: 
            for point in polygon: 
                if point[0] < minX: 
                    minX = point[0]
                if point[1] < minY: 
                    minY = point[1]
                if point[0] > maxX: 
                    maxX = point[0]
                if point[1] > maxY: 
                    maxY = point[1]
        return minX, minY, maxX, maxY

    def move(self, deltaX, deltaY):
        for polygon in self._polygons: 
            for point in polygon: 
                point[0] += deltaX
                point[1] += deltaY

    def center(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512): 
        canvas = self.range()
        canvasX = canvas[2] - canvas[0]
        canvasY = canvas[3] - canvas[1]
        halfX = (sizeX - canvasX) // 2
        halfY = (sizeY - canvasY) // 2
        deltaX = halfX - canvas[0]
        deltaY = halfY - canvas[1]
        self.move(deltaX - offsetX, deltaY - offsetY)

    def image(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512): 
        polygons = list(map(lambda x: np.array(x, np.int64) + np.array([[offsetX, offsetY]]), self._polygons))
        img = np.zeros([sizeX, sizeY], dtype=np.float32)
        for idx in range(len(polygons)): 
            img = cv2.fillPoly(img, [polygons[idx]], color=255)
        return img

    def mat(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512): 
        return self.image(sizeX, sizeY, offsetX, offsetY) / 255.0

    def export(self, filename): 
        with open(filename, "w") as fout: 
            fout.write(f"BEGIN     /* The metadata are invalid */\n")
            fout.write(f"EQUIV  1  1000  MICRON  +X,+Y\n")
            fout.write(f"CNAME Temp_Top\n")
            fout.write(f"LEVEL M1\n")
            fout.write(f"\n")
            fout.write(f"CELL Temp_Top PRIME\n")
            for kdx, polygon in enumerate(self._polygons): 
                info = ""
                for point in polygon: 
                    info += " " + str(point[0]) + " " + str(point[1])
                fout.write(f"   PGON N M1 {info}\n")
            fout.write(f"ENDMSG\n")

    def split(self, sizeX=16384, sizeY=16384, strideX=4096, strideY=4096, write=True): 
        minX, minY = 1e12, 1e12
        maxX, maxY = -1e12, -1e12
        ranges = []
        for polygon in self._polygons: 
            minXpoly, minYpoly = 1e12, 1e12
            maxXpoly, maxYpoly = -1e12, -1e12
            for coord in polygon: 
                if coord[0] > maxX: 
                    maxX = coord[0]
                if coord[1] > maxY: 
                    maxY = coord[1]
                if coord[0] < minX: 
                    minX = coord[0]
                if coord[1] < minY: 
                    minY = coord[1]
                if coord[0] > maxXpoly: 
                    maxXpoly = coord[0]
                if coord[1] > maxYpoly: 
                    maxYpoly = coord[1]
                if coord[0] < minXpoly: 
                    minXpoly = coord[0]
                if coord[1] < minYpoly: 
                    minYpoly = coord[1]
            ranges.append([minXpoly, minYpoly, maxXpoly, maxYpoly])
        print(f"[Design.split]: range ({minX, minY}) -> ({maxX, maxY})")

        intervalX = maxX - minX
        intervalY = maxY - minY
        stepsX = round((intervalX - (sizeX - strideX)) / strideX)
        stepsY = round((intervalY - (sizeY - strideY)) / strideY)
        print(f"[Design.split]: tiles ({stepsX, stepsY})")

        offsets = [[(None, None) for _ in range(stepsY)] for _ in range(stepsY)]
        visited = [False for _ in range(len(self._polygons))]

        for idx in range(stepsX): 
            for jdx in range(stepsY): 
                startX = minX + idx * strideX
                startY = minY + jdx * strideY
                endX = startX + sizeX
                endY = startY + sizeY
                polygons = []
                for kdx, polygon in enumerate(self._polygons): 
                    # if ranges[kdx][0] >= endX or ranges[kdx][1] >= endY or ranges[kdx][2] < startX or ranges[kdx][3] < startY: 
                    if ranges[kdx][0] >= startX and ranges[kdx][1] >= startY and ranges[kdx][2] < endX and ranges[kdx][3] < endY: 
                        polygons.append(polygon)
                        visited[kdx] = True
                offset = (startX, startY)
                if write: 
                    filename = self._filename[:-4] + f"__{idx}_{jdx}" + ".glp"
                    with open(filename, "w") as fout: 
                        fout.write(f"BEGIN     /* The metadata are invalid */\n")
                        fout.write(f"EQUIV  1  1000  MICRON  +X,+Y\n")
                        fout.write(f"CNAME Temp_Top\n")
                        fout.write(f"LEVEL M1\n")
                        fout.write(f"\n")
                        fout.write(f"CELL Temp_Top PRIME\n")
                        for polygon in polygons: 
                            info = ""
                            for point in polygon: 
                                info += " " + str(point[0]-offset[0]) + " " + str(point[1]-offset[1])
                            fout.write(f"   PGON N M1 {info}\n")
                        fout.write(f"ENDMSG\n")
        countCross = 0
        for kdx, polygon in enumerate(self._polygons): 
            if not visited[kdx]: 
                countCross += 1
        if write: 
            filename = self._filename[:-4] + f"__cross" + ".glp"
            with open(filename, "w") as fout: 
                fout.write(f"BEGIN     /* The metadata are invalid */\n")
                fout.write(f"EQUIV  1  1000  MICRON  +X,+Y\n")
                fout.write(f"CNAME Temp_Top\n")
                fout.write(f"LEVEL M1\n")
                fout.write(f"\n")
                fout.write(f"CELL Temp_Top PRIME\n")
                for kdx, polygon in enumerate(self._polygons): 
                    if not visited[kdx]: 
                        info = ""
                        for point in polygon: 
                            info += " " + str(point[0]) + " " + str(point[1])
                        fout.write(f"   PGON N M1 {info}\n")
                fout.write(f"ENDMSG\n")
        
        return countCross
                    

        
        

if __name__ == "__main__": 
    # design = Design("./benchmark/ICCAD2013/M1_test1.glp")
    # design.image()

    # design = Design("tmp/gcd.glp")
    # print(f"Range: {design.range()}")
    # size = design.range()
    # sizeILT = int(math.ceil(max(size)/2048) * 2048)
    # img = design.image(sizeX=sizeILT, sizeY=sizeILT, offsetX=0, offsetY=0)
    # cv2.imwrite("tmp/tmp.png", img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()

    # design = Design("work/gds_diff/gcd/gcd.glp")
    design = Design("work/gds_diff/aes/aes_cipher_top.glp")
    count = design.split(sizeX=65536, sizeY=65536, strideX=16384, strideY=16384, write=True)
    print(count)
