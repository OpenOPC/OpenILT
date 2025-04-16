import copy
import logging

import torch

from mtilt.solving import build_initializer


__all__ = ["BenchmarkMapper"]

logger = logging.getLogger(__name__)

class BenchmarkMapper():


    def __init__(self, cfg):

        self.initializer = build_initializer(cfg)
        # todo: add some logging info here
        logger.info(f"You are using {cfg.INITIALIZER.NAME} to initialize your mask.")

        self.cfg = cfg
        self.tilesizeX = cfg.DESIGN.TILE_SIZE_X
        self.tilesizeY = cfg.DESIGN.TILE_SIZE_Y
        self.offsetX   = cfg.DESIGN.OFF_SET_X
        self.offsetY   = cfg.DESIGN.OFF_SET_Y
        self.iltsizeX  = cfg.DESIGN.ILT_SIZE_X
        self.iltsizeY  = cfg.DESIGN.ILT_SIZE_Y

        self.scale = cfg.DESIGN.SCALE

    def __call__(self, dataset_dict):

        design = dataset_dict['design_ins']
        ref = copy.deepcopy(dataset_dict['design_ins'])

        design.polygons = self.scale
        design.center(self.tilesizeX, self.tilesizeY, self.offsetX, self.offsetY)
        params, target = self.initializer(design)

        ref.polygons = 1
        ref.center(self.tilesizeX * self.scale, self.tilesizeY * self.scale,
                   self.offsetX * self.scale, self.offsetY * self.scale)
        _, ref_target = self.initializer(ref, self.scale)

        dataset_dict.pop('design_ins')

        dataset_dict["backup"] = params
        dataset_dict["params"] = params.clone().detach().requires_grad_(True)
        dataset_dict["target"] = target
        dataset_dict["ref_target"] = ref_target

        return dataset_dict

