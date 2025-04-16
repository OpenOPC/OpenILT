import multiprocessing as mp

import torch
import torch.nn as nn
import numpy as np


from mtilt.config import configurable, CfgNode
from mtilt.utils.events import get_event_storage
from .build import META_INITIALIZER_REGISTRY

@META_INITIALIZER_REGISTRY.register()
class LevelSetInitializer():

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
        self.offsetX   = offsetX
        self.offsetY   = offsetY

        self.device = device
        self.dtype = dtype

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        if value is not None:
            self._filter = value
        else:
            raise ValueError("invalid value for a torch.Tensor format data.")

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

    def _distMatPolygon(self, polygon, canvas, offsets):
        if len(canvas) == 4:
            canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
        minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
        sizeX, sizeY = maxX - minX, maxY - minY

        dist = np.ones([sizeX, sizeY]) * (sizeX * sizeY)
        xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
        ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
        xs = np.tile(xs, [1, sizeY])
        ys = np.tile(ys, [sizeX, 1])

        frPt = polygon[-1]
        for toPt in polygon:
            frX, frY = frPt
            toX, toY = toPt
            if frX > toX:
                frX, toX = toX, frX
            if frY > toY:
                frY, toY = toY, frY
            frX += offsets[0]
            toX += offsets[0]
            frY += offsets[1]
            toY += offsets[1]

            dist1 = np.sqrt((frX - xs) ** 2 + (frY - ys) ** 2)
            dist2 = np.sqrt((toX - xs) ** 2 + (toY - ys) ** 2)

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
        return dist.T

    def _distMatLegacy(self, design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]):
        if len(canvas) == 4:
            canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
        minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

        mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
        dist = np.ones([maxX - minX, maxY - minY]) * ((maxX - minX) * (maxY - minY))
        for polygon in design.polygons:
            tmp = self._distMatPolygon(polygon, canvas, offsets)
            dist = np.minimum(dist, tmp)
        dist[mask > 0] *= -1
        return dist

    def _distMatPolygonTorch(self, polygon, canvas, offsets):
        if len(canvas) == 4:
            canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
        minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
        sizeX, sizeY = maxX - minX, maxY - minY

        dist = torch.ones([sizeX, sizeY], dtype=self.dtype, device=self.device) * (sizeX * sizeY)
        xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
        ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
        xs = torch.tensor(np.tile(xs, [1, sizeY]), dtype=self.dtype, device=self.device)
        ys = torch.tensor(np.tile(ys, [sizeX, 1]), dtype=self.dtype, device=self.device)

        frPt = polygon[-1]
        for toPt in polygon:
            frX, frY = frPt
            toX, toY = toPt
            if frX > toX:
                frX, toX = toX, frX
            if frY > toY:
                frY, toY = toY, frY
            frX += offsets[0]
            toX += offsets[0]
            frY += offsets[1]
            toY += offsets[1]

            dist1 = torch.sqrt((frX - xs) ** 2 + (frY - ys) ** 2)
            dist2 = torch.sqrt((toX - xs) ** 2 + (toY - ys) ** 2)

            dist = torch.minimum(dist, torch.minimum(dist1, dist2))

            if frX == toX:
                mask = (frY <= ys) * (ys <= toY)
                new = torch.minimum(dist, torch.abs(frX - xs))
                dist[mask] = new[mask]
            elif frY == toY:
                mask = (frX <= xs) * (xs <= toX)
                new = torch.minimum(dist, torch.abs(frY - ys))
                dist[mask] = new[mask]

            frPt = toPt
        return dist.T

    def _distMatTorch(self, design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512], mask=None):
        if len(canvas) == 4:
            canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
        minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

        if mask is None:
            mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
        dist = torch.ones([maxX - minX, maxY - minY],
                          dtype=self.dtype, device=self.device) * ((maxX - minX) * (maxY - minY))
        for polygon in design.polygons:
            tmp = self._distMatPolygonTorch(polygon, canvas, offsets)
            dist = torch.minimum(dist, tmp)
        dist[mask > 0] *= -1
        return dist

    def _distMat(self, design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]):
        if len(canvas) == 4:
            canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
        minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

        pool = mp.Pool(processes=mp.cpu_count() // 2)
        procs = []
        for polygon in design.polygons:
            proc = pool.apply_async(self._distMatPolygon, (polygon, canvas, offsets))
            procs.append(proc)
        pool.close()
        pool.join()

        dist = np.ones([maxX- minX, maxY - minY]) * ((maxX - minX) * (maxY - minY))
        for proc in procs:
            tmp = proc.get()
            dist = np.minimum(dist, tmp)
        mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
        dist[mask > 0] *= -1

        return dist

    def __call__(self, design, scale=1):

        target = torch.tensor(design.mat(self.tilesizeX*scale, self.tilesizeY*scale,
                                         self.offsetX*scale, self.offsetY*scale),
                              dtype=self.dtype, device=self.device)
        params = self._distMatTorch(design, canvas=[[0, 0], [self.tilesizeX*scale, self.tilesizeY*scale]],
                               offsets=[self.offsetX*scale, self.offsetY*scale]).detach().clone().requires_grad_(True)
        return params, target