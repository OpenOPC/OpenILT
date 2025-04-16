import torch
import torch.nn as nn

from mtilt.config import configurable, CfgNode
from mtilt.utils.events import get_event_storage
from .build import META_INITIALIZER_REGISTRY

@META_INITIALIZER_REGISTRY.register()
class PixelInitializer():

    @configurable
    def __init__(
            self,
            config: CfgNode,
            tilesizeX,
            tilesizeY,
            offsetX,
            offsetY,
            device="cuda:0",
            dtype=torch.float,
    ):

        self.config = config
        self.tilesizeX = tilesizeX
        self.tilesizeY = tilesizeY
        self.offsetX = offsetX
        self.offsetY = offsetY

        self.device = device
        self.dtype = dtype


    @classmethod
    def from_config(cls, cfg):

        return {"config": cfg,
                "tilesizeX": cfg.DESIGN.TILE_SIZE_X,
                "tilesizeY": cfg.DESIGN.TILE_SIZE_Y,
                "offsetX":   cfg.DESIGN.OFF_SET_X,
                "offsetY":   cfg.DESIGN.OFF_SET_Y,
                "device":    cfg.LITHO_OPERATOR.DEVICE,
                "dtype":     eval(cfg.REALTYPE),
        }

    def __call__(self, design, scale=1):

        design = design.mat(self.tilesizeX*scale, self.tilesizeY*scale, self.offsetX*scale, self.offsetY*scale)
        target = torch.tensor(design, dtype=self.dtype, device=self.device)
        params = target * 2.0 - 1.0
        return params, target