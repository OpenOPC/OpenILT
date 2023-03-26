import copy
import math

USECV2 = False
try: 
    import cv2
    USECV2 = True
except Exception: 
    pass
import numpy as np
import matplotlib.pyplot as plt

from gdsii.library import *
from gdsii.structure import *
from gdsii.elements import *

class PolygonGDS: 
    def __init__(self, element=None, layer=None, points=None): 
        self._element = element
        self._layer = None
        self._points = []
        if self._element is None: 
            self._layer = layer
            self._points = points
        elif isinstance(self._element, Boundary): 
            self._layer = self._element.layer
            self._points = self._element.xy
            if self._points[0][0] == self._points[-1][0] and self._points[0][1] == self._points[-1][1]: 
                self._points = self._points[:-1]
            for idx in range(len(self._points)): 
                self._points[idx] = list(self._points[idx])
        elif isinstance(self._element, Path): 
            self._layer = self._element.layer
            halfW = self._element.width // 2
            points0 = self._element.xy
            points1 = []
            points2 = []
            typeLast = None
            for idx in range(len(points0) - 1): # don't forget to deal with the last one
                frX, frY = points0[idx]
                toX, toY = points0[idx+1]
                if frX == toX: # vertical
                    if frY > toY: # downward
                        if typeLast is None: 
                            points1.append([frX - halfW, frY])
                            points2.append([frX + halfW, frY])
                        elif typeLast == "left": 
                            points1.append([frX - halfW, frY - halfW])
                            points2.append([frX + halfW, frY + halfW])
                        elif typeLast == "right": 
                            points1.append([frX - halfW, frY + halfW])
                            points2.append([frX + halfW, frY - halfW])
                        elif typeLast == "down": 
                            pass
                        else: 
                            print(self._element.xy)
                            assert False, f"Unsupported direction! Last direction is {typeLast}"
                        typeLast = "down"
                        
                        if idx + 1 == len(points0) - 1: 
                            points1.append([toX - halfW, toY])
                            points2.append([toX + halfW, toY])
                    else: # upward
                        if typeLast is None: 
                            points1.append([frX + halfW, frY])
                            points2.append([frX - halfW, frY])
                        elif typeLast == "left": 
                            points1.append([frX + halfW, frY - halfW])
                            points2.append([frX - halfW, frY + halfW])
                        elif typeLast == "right": 
                            points1.append([frX + halfW, frY + halfW])
                            points2.append([frX - halfW, frY - halfW])
                        elif typeLast == "up": 
                            pass
                        else: 
                            print(self._element.xy)
                            assert False, f"Unsupported direction! Last direction is {typeLast}"
                        typeLast = "up"
                        
                        if idx + 1 == len(points0) - 1: 
                            points1.append([toX + halfW, toY])
                            points2.append([toX - halfW, toY])
                elif frY == toY: # horizontal
                    if frX > toX: # leftward
                        if typeLast is None: 
                            points1.append([frX, frY - halfW])
                            points2.append([frX, frY + halfW])
                        elif typeLast == "down": 
                            points1.append([frX - halfW, frY - halfW])
                            points2.append([frX + halfW, frY + halfW])
                        elif typeLast == "up": 
                            points1.append([frX + halfW, frY - halfW])
                            points2.append([frX - halfW, frY + halfW])
                        elif typeLast == "left": 
                            pass
                        else: 
                            print(self._element.xy)
                            assert False, f"Unsupported direction! Last direction is {typeLast}"
                        typeLast = "left"
                        
                        if idx + 1 == len(points0) - 1: 
                            points1.append([toX, toY - halfW])
                            points2.append([toX, toY + halfW])
                    else: # rightward
                        if typeLast is None: 
                            points1.append([frX, frY + halfW])
                            points2.append([frX, frY - halfW])
                        elif typeLast == "down": 
                            points1.append([frX - halfW, frY + halfW])
                            points2.append([frX + halfW, frY - halfW])
                        elif typeLast == "up": 
                            points1.append([frX + halfW, frY + halfW])
                            points2.append([frX - halfW, frY - halfW])
                        elif typeLast == "right": 
                            pass
                        else: 
                            print(self._element.xy)
                            assert False, f"Unsupported direction! Last direction is {typeLast}"
                        typeLast = "right"
                        
                        if idx + 1 == len(points0) - 1: 
                            points1.append([toX, toY + halfW])
                            points2.append([toX, toY - halfW])
                else: 
                    assert False, "Unsupported path!"
            assert len(points1) == len(points2)
            self._points = []
            for idx in range(len(points1)): 
                self._points.append(points1[idx])
            for idx in range(len(points2)): 
                self._points.append(points2[-1-idx])

    @property
    def layer(self): 
        return self._layer

    @property
    def points(self): 
        return self._points.copy()

    def __repr__(self): 
        return f"(Polygon {self._points})"

    def pointIn(self, point): 
        isOdd = False
        frPt = self._points[-1]
        for toPt in self._points: 
            frX, frY = frPt
            toX, toY = toPt
            if (((frY < point[1] and point[1] <= toY) or (toY < point[1] and point[1] <= frY)) and \
               (frX + (point[1] - frY) / (toY - frY) * (toX - frX) < point[0])): 
               isOdd = not isOdd
            frPt = toPt
        return isOdd

    def pointOn(self, point): 
        frPt = self._points[-1]
        for toPt in self._points: 
            frX, frY = frPt
            toX, toY = toPt
            if frX > toX: 
                frX, toX = toX, frX
            if frY > toY: 
                frY, toY = toY, frY
            if (frX == toX == point[0] and frY <= point[1] <= toY) or \
               (frY == toY == point[1] and frX <= point[0] <= toX): 
               return True
            frPt = toPt
        return False

    def distance(self, point): 
        if self.pointOn(point): 
            return 0.0

        dist = 1e15
        frPt = self._points[-1]
        for toPt in self._points: 
            frX, frY = frPt
            toX, toY = toPt
            if frX > toX: 
                frX, toX = toX, frX
            if frY > toY: 
                frY, toY = toY, frY
            
            if (frX == toX) and (frY <= point[1] <= toY): 
                dist = min(dist, abs(frX - point[0]))
            elif (frY == toY) and (frX <= point[0] <= toX): 
                dist = min(dist, abs(frY - point[1]))
            dist = min(dist, math.sqrt((frX - point[0])**2 + (frY - point[1])**2))
            dist = min(dist, math.sqrt((toX - point[0])**2 + (toY - point[1])**2))
                
            frPt = toPt
        return dist

    def distMat(self, canvas): 
        if len(canvas) == 4: 
            canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
        minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
        sizeX, sizeY = maxX - minX, maxY - minY

        dist = np.ones([sizeX, sizeY]) * (sizeX * sizeY)
        xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
        ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
        xs = np.tile(xs, [1, sizeY])
        ys = np.tile(ys, [sizeX, 1])
        
        frPt = self._points[-1]
        for toPt in self._points: 
            frX, frY = frPt
            toX, toY = toPt
            if frX > toX: 
                frX, toX = toX, frX
            if frY > toY: 
                frY, toY = toY, frY
            
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
        return dist

    def getRects(self): 
        rects = []
        hLevels = set()
        vLevels = set()
        frPt = self._points[-1]
        for toPt in self._points: 
            frX, frY = frPt
            toX, toY = toPt
            if frX == toX: 
                vLevels.add(frX)
            elif frY == toY: 
                hLevels.add(frY)
            else: 
                assert False, "[PolygonGDS]: Diagonal edge not supported!"
            frPt = toPt
        hLevels = sorted(list(hLevels))
        vLevels = sorted(list(vLevels))
        for idx in range(len(vLevels) - 1): 
            for jdx in range(len(hLevels) - 1): 
                corner1 = [vLevels[idx],   hLevels[jdx]]
                corner2 = [vLevels[idx+1], hLevels[jdx+1]]
                center  = [(corner1[0] + corner2[0]) / 2.0, (corner1[1] + corner2[1]) / 2.0]
                if self.pointIn(center): 
                    assert corner1[0] < corner2[0] and corner1[1] < corner2[1]
                    rects.append([corner1, corner2])
        return rects

class SRefGDS: 
    flatLib = {}

    def __init__(self, struct, xy, strans=None, mag=None, angle=None): 
        self._struct = struct
        self._xy = xy[0] if len(xy) == 1 else xy
        self._reflect = False
        if strans is None or strans == 0: 
            pass
        elif strans == 32768: 
            self._reflect = True
        else: 
            assert False, f"Unsupported strans = {strans}"
        self._scale = 1.0
        if mag == None: 
            pass
        else: 
            self._scale = mag
        self._rotate = 0.0
        if angle is None: 
            pass
        elif angle in [0.0, 90.0, 180.0, 270.0]: 
            self._rotate = angle
        else: 
            assert False, f"[SRefGDS]: Unsupported rotate angle = {self._rotate}"
    
    def flatten(self): 
        # print(f"[Flattening] a SRef of {self._struct.name}")
        libname = self._struct._name + f"__FLAT({self._reflect, self._scale, self._rotate})"
        if libname in SRefGDS.flatLib: 
            # print(f" -> skip the construction of sref {libname}")
            struct = copy.deepcopy(SRefGDS.flatLib[libname])
            for element in struct._elements: 
                for point in element.points: 
                    point[0] = point[0] + self._xy[0]
                    point[1] = point[1] + self._xy[1]
            return struct

        elements = [] 
        layers = copy.deepcopy(self._struct._layers)
        for element in self._struct._elements: 
            elements.append(copy.deepcopy(element))
        if len(self._struct._srefs) > 0: 
            for sref in self._struct._srefs: 
                struct = sref.flatten()
                elements.extend(copy.deepcopy(struct._elements))
                layers.extend(copy.deepcopy(struct._layers))
                layers = list(set(layers))

        if self._reflect: 
            for element in elements: 
                for point in element.points: 
                    point[1] = -point[1]
        if self._scale != 1.0: 
            for element in elements: 
                for point in element.points: 
                    tmp0 = point[0] * self._scale
                    tmp1 = point[1] * self._scale
                    point[0] = int(round(tmp0))
                    point[1] = int(round(tmp1))
        if self._rotate != 0.0: 
            if self._rotate == 90.0: 
                for element in elements: 
                    for point in element.points: 
                        tmp0 = math.cos(self._rotate) * point[0] - math.sin(self._rotate) * point[1]
                        tmp1 = math.cos(self._rotate) * point[1] + math.sin(self._rotate) * point[0]
                        point[0] = -point[1]
                        point[1] = point[0]
            elif self._rotate == 180.0: 
                for element in elements: 
                    for point in element.points: 
                        point[0] = -point[0]
                        point[1] = -point[1]
            elif self._rotate == 270.0: 
                for element in elements: 
                    for point in element.points: 
                        point[0] = point[1]
                        point[1] = -point[0]

        struct = StructGDS()
        struct._name = libname
        struct._elements = elements
        struct._layers = layers
        SRefGDS.flatLib[struct._name] = copy.deepcopy(struct)
        for element in struct._elements: 
            for point in element.points: 
                point[0] = point[0] + self._xy[0]
                point[1] = point[1] + self._xy[1]

        #TODO translation!!!!
        return struct


class StructGDS: 
    def __init__(self, struct=None, data=None): 
        self._struct = struct
        if self._struct is None: 
            self._name = None
            self._elements = []
            self._layers = []
            self._srefs = []
            return
        self._name = self._struct.name.decode()
        self._elements = []
        self._layers = set()
        self._srefs = []
        # print("[Structure]", self._name)
        for idx in range(len(self._struct)): 
            if isinstance(self._struct[idx], Boundary): 
                polygon = PolygonGDS(self._struct[idx])
                self._elements.append(polygon)
                self._layers.add(polygon.layer)
                # print("[Boundary]", polygon.points)
            elif isinstance(self._struct[idx], Path): 
                polygon = PolygonGDS(self._struct[idx])
                self._elements.append(polygon)
                self._layers.add(polygon.layer)
                # print("[Path]", self._struct[idx].xy, self._struct[idx].width)
                # print(" -> ", polygon.points)
            elif isinstance(self._struct[idx], SRef): 
                assert not data is None
                name = self._struct[idx].struct_name.decode()
                assert name in data, f"[StructGDS]: Cannot find defined structure = {name}"
                struct = data[name]
                points = self._struct[idx].xy
                strans = self._struct[idx].strans
                mag = self._struct[idx].mag
                angle = self._struct[idx].angle
                self._srefs.append(SRefGDS(struct, points, strans, mag, angle))
            elif isinstance(self._struct[idx], ARef): 
                assert not isinstance(self._struct[idx], ARef) # Do not support ARef
            elif isinstance(self._struct[idx], Box): 
                assert not isinstance(self._struct[idx], Box) # Do not support Box
            elif isinstance(self._struct[idx], Node): 
                assert not isinstance(self._struct[idx], Node) # Do not support Node
            elif isinstance(self._struct[idx], Text): 
                continue # Skip the texts
        for idx in range(len(self._struct)): 
            pass
        self._layers = list(self._layers)

    @property
    def name(self): 
        return self._name

    @property
    def elements(self): 
        return self._elements

    @property
    def layers(self): 
        return self._layers
    
    @property
    def hasSRef(self): 
        return len(self._srefs) > 0

    def flatten(self): 
        return SRefGDS(self, xy=[(0, 0)], strans=None, mag=None, angle=None).flatten()

    def polygons(self, layers=None): 
        polygons = []
        for element in self._elements: 
            if layers is None or element.layer in layers: 
                polygons.append(element)
        return polygons

    def exportGLP(self, filename, scale=1, layers=None): 
        minX = 1e12
        minY = 1e12
        valid = False
        for polygon in self.polygons(): 
            if layers is None or polygon.layer in layers: 
                valid = True
                for point in polygon.points: 
                    if point[0] < minX: 
                        minX = point[0]
                    if point[1] < minY: 
                        minY = point[1]
        if not valid: 
            print(f"[ReadGDS]: Skip an empty structure for specifed layers, {filename}")
            return None, None
        print(f"[ReadGDS]: Dumping it to {filename}")    
        with open(filename, "w") as fout: 
            fout.write(f"BEGIN     /* The metadata are invalid */\n")
            fout.write(f"EQUIV  1  1000  MICRON  +X,+Y\n")
            fout.write(f"CNAME Temp_Top\n")
            fout.write(f"LEVEL M1\n")
            fout.write(f"\n")
            fout.write(f"CELL Temp_Top PRIME\n")
            for polygon in self.polygons(): 
                if layers is None or polygon.layer in layers: 
                    info = ""
                    for point in polygon.points: 
                        info += " " + str(round((point[0]-minX)*scale)) + " " + str(round((point[1]-minY)*scale))
                    fout.write(f"   PGON N M1 {info}\n")
            fout.write(f"ENDMSG\n")
        return minX, minY
        
    def image(self, scale=1, padding=0, layers=None, center=False): 
        if len(self._srefs) > 0: 
            print("[StructGDS]: Unable to draw it so far. ")
            return
        polygons = []
        for element in self._elements: 
            if layers is None or element.layer in layers: 
                polygons.append(element.points)
        if len(polygons) == 0: 
            print("[StructGDS]: Skip an empty structure for specifed layers")
            return
        print(f"[StructGDS]: In total, {len(polygons)} polygons")
        minX, minY = 1e15, 1e15
        maxX, maxY = -1e15, -1e15
        for polygon in polygons: 
            points = np.array(polygon)
            mins = np.min(polygon, axis=0)
            maxs = np.max(polygon, axis=0)
            minX = min(minX, mins[0])
            minY = min(minY, mins[1])
            maxX = max(maxX, maxs[0])
            maxY = max(maxY, maxs[1])
        print(f"[StructGDS]: Polygons range: {minX, minY}, {maxX, maxY}")
        if center: 
            maxX = max(abs(minX), abs(maxX))
            maxY = max(abs(minY), abs(maxY))
            minX = -maxX
            minY = -maxY
        for polygon in polygons: 
            for idx in range(len(polygon)): 
                polygon[idx] = list(polygon[idx])
        for polygon in polygons: 
            for point in polygon: 
                point[0] -= minX
                point[1] -= minY
                point[0] = int(point[0] * scale)
                point[1] = int(point[1] * scale)
                point[0] += padding
                point[1] += padding
        minX, minY = 1e15, 1e15
        maxX, maxY = -1e15, -1e15
        for polygon in polygons: 
            points = np.array(polygon)
            mins = np.min(polygon, axis=0)
            maxs = np.max(polygon, axis=0)
            minX = min(minX, mins[0])
            minY = min(minY, mins[1])
            maxX = max(maxX, maxs[0])
            maxY = max(maxY, maxs[1])
        print(f" -> Polygons range: {minX, minY}, {maxX, maxY}")
        sizeDrawX = maxX + padding
        sizeDrawY = maxY + padding
        image = np.zeros([sizeDrawX, sizeDrawY], dtype=np.uint8)
        if USECV2: 
            cv2.fillPoly(image, list(map(lambda x: np.array(list(map(lambda y: [y[1], y[0]], x))), polygons)), color=[255])
            return image
        for idx, polygon in enumerate(polygons): 
            print(f"\r -> Painting No.{idx}/{len(polygons)}", end="")
            rects = PolygonGDS(points=polygon).getRects()
            # print(rects)
            for rect in rects: 
                image[rect[0][0]:(rect[1][0]+1), rect[0][1]:(rect[1][1]+1)] = 1
        print()
        return image
        
    def draw(self, scale=1, padding=0, layers=None, center=False): 
        image = self.image(scale, padding, layers, center)
        if not image is None: 
            plt.imshow(image)
            plt.show()


class ReaderGDS: 
    def __init__(self, filename:str): 
        self._filename = filename
        with open(self._filename, "rb") as stream:
            self._library = Library.load(stream)
        self._structures = {}
        print(f"[GDSReader]: Reading GDS file: {self._filename}")
        print(" -> Name:", self._library.name)
        print(" -> Physical Unit:", self._library.physical_unit)
        print(" -> Logical Unit:", self._library.logical_unit)
        self._name = self._library.name
        self._unitPhy = self._library.physical_unit
        self._unitUser = self._library.logical_unit
        self._unit = self._unitPhy
        for idx in range(len(self._library)): 
            struct = StructGDS(self._library[idx], self._structures)
            self._structures[struct.name] = struct
        layers = set()
        for name, struct in self._structures.items(): 
            for element in struct.elements: 
                layers.add(element.layer)
        self._layers = list(layers)
        print(" -> Layers:", self._layers)
    
    @property
    def unit(self): 
        return self._unit
    @property
    def structs(self): 
        return self._structures

    def draw(self, scale=1.0, padding=0, layers=None, center=False, noSRef=False): 
        for name, struct in self._structures.items(): 
            if len(struct._srefs) > 0: 
                flattened = struct.flatten()
                flattened.draw(scale, padding, layers, center)
            elif noSRef: 
                struct.draw(scale, padding, layers, center)

    def haveSref(self): 
        result = []
        for name, struct in self._structures.items(): 
            if len(struct._srefs) > 0: 
                result.append(name)

    def polygons(self, name=None, layers=None): 
        if not name is None: 
            struct = self._structures[name]
            return struct.flatten().polygons(layers) if struct.hasSRef() else struct.polygons(layers)
        elif len(self._structures) == 1: 
            name = list(self._structures.keys())[0]
            struct = self._structures[name]
            return struct.polygons(layers)
        else: 
            count = 0
            name = None
            for key, struct in self._structures.items(): 
                if struct.hasSRef(): 
                    count += 1
                    name = key
            if count == 1: 
                struct = self._structures[name]
                return struct.flatten().polygons(layers) if struct.hasSRef() else struct.polygons(layers)
            else: 
                assert count == 1, f"Expect only one structure that contains SRefs, but get {count}. Please specify the structure to use. "

if __name__ == "__main__": 
    # reader = ReaderGDS("s.gds")
    # reader.draw(0.01, 0, layers=(10001, ), center=False, noSRef=True)
    # reader = ReaderGDS("tmp/layout.gds")
    # reader.draw(0.02, 200, layers=(10, ), center=False, noSRef=False)

    basedir = "tmp/gds_diff/gcd/"
    filename = "6_final.gds"
    reader = ReaderGDS(basedir + filename)
    structs = reader.structs
    for name, struct in structs.items(): 
        print(f"[Structure]: {name}")
        if len(struct._srefs) > 0: 
            print(f"SRAF encountered: {name}")
            struct = struct.flatten()
            # struct.draw(scale=0.1, layers=(1, ))
            struct.exportGLP(f"{basedir}/{name}.glp", scale=0.1, layers=(1, ))
        else: 
            # struct.draw(layers=(1, ))
            struct.exportGLP(f"{basedir}/{name}.glp", scale=0.1, layers=(1, ))


