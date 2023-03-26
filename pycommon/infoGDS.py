import sys
import numpy as np

from gdsii.library import Library
from gdsii.elements import *

# read a library from a file
assert len(sys.argv) > 1, f"A GDS file must be specified"
with open(sys.argv[1], "rb") as stream:
    lib = Library.load(stream)

print("name:", lib.name)
print("physical_unit:", lib.physical_unit)
print("logical_unit:", lib.logical_unit)
for idx in range(len(lib)): 
    print("[Structure]", lib[idx].name.decode())
    for jdx in range(len(lib[idx])): 
        if isinstance(lib[idx][jdx], Boundary): 
            print(" -> [Boundary]")
            print(" ->  -> layer", lib[idx][jdx].layer)
            print(" ->  -> data_type", lib[idx][jdx].data_type)
            print(" ->  -> xy", lib[idx][jdx].xy)

            points = lib[idx][jdx].xy
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            print(" ->  -> size", (maxs[0] - mins[0]), "x", (maxs[1] - mins[1]))
        elif isinstance(lib[idx][jdx], Path): 
            print(" -> [Path]")
            print(" ->  -> layer", lib[idx][jdx].layer)
            print(" ->  -> data_type", lib[idx][jdx].data_type)
            print(" ->  -> xy", lib[idx][jdx].xy)
            print(" ->  -> width", lib[idx][jdx].width)
            print(" ->  -> path_type", lib[idx][jdx].path_type) # 0: rectangle, 1: semi-circle, 2: some wierd type

            points = lib[idx][jdx].xy
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            print(" ->  -> size", (maxs[0] - mins[0]), "x", (maxs[1] - mins[1]))
        elif isinstance(lib[idx][jdx], SRef): 
            print(" -> [SRef]")
            print(" ->  -> struct_name", lib[idx][jdx].struct_name.decode())
            print(" ->  -> xy", lib[idx][jdx].xy)
            print(" ->  -> strans", lib[idx][jdx].strans) # mirrow
            print(" ->  -> mag", lib[idx][jdx].mag) # scale
            print(" ->  -> angle", lib[idx][jdx].angle) # rotate
        elif isinstance(lib[idx][jdx], ARef): 
            assert not isinstance(lib[idx][jdx], ARef) # Do not support ARef
            print(" -> [ARef]")
            print(" ->  -> struct_name", lib[idx][jdx].struct_name.decode())
            print(" ->  -> xy", lib[idx][jdx].xy)
            print(" ->  -> cols", lib[idx][jdx].cols)
            print(" ->  -> rows", lib[idx][jdx].rows)
            print(" ->  -> strans", lib[idx][jdx].strans)
            print(" ->  -> mag", lib[idx][jdx].mag)
            print(" ->  -> angle", lib[idx][jdx].angle)
        elif isinstance(lib[idx][jdx], Box): 
            assert not isinstance(lib[idx][jdx], Box) # Do not support Box
            print(" -> [Box]")
            print(" ->  -> layer", lib[idx][jdx].layer)
            print(" ->  -> box_type", lib[idx][jdx].box_type)
            print(" ->  -> xy", lib[idx][jdx].xy)
        elif isinstance(lib[idx][jdx], Node): 
            assert not isinstance(lib[idx][jdx], Node) # Do not support Node
            print(" -> [Node]")
            print(" ->  -> layer", lib[idx][jdx].layer)
            print(" ->  -> node_type", lib[idx][jdx].node_type)
            print(" ->  -> xy", lib[idx][jdx].xy)
        elif isinstance(lib[idx][jdx], Text): 
            continue # Ignore the texts
            print(" -> [Text]")
            print(" ->  -> layer", lib[idx][jdx].layer)
            print(" ->  -> text_type", lib[idx][jdx].text_type)
            print(" ->  -> xy", lib[idx][jdx].xy)
            print(" ->  -> string", lib[idx][jdx].string)