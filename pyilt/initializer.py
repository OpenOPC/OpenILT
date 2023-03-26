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