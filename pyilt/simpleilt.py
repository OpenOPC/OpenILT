import sys
sys.path.append(".")
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pylitho.simple as lithosim
# import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

class SimpleCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]

class SimpleILT: 
    def __init__(self, config=SimpleCfg("./config/simpleilt2048.txt"), lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False): 
        super(SimpleILT, self).__init__()
        self._config = config
        self._device = device
        # Lithosim
        self._lithosim = lithosim.to(DEVICE)
        if multigpu: 
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
    
    def solve(self, target, params, curv=None, verbose=0): 
        # Initialize
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor): 
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        # opt = optim.SGD([params], lr=1.6e0)
        opt = optim.Adam([params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._config["Iterations"]): 
            mask = torch.sigmoid(self._config["SigmoidSteepness"] * params) * self._filter
            mask += torch.sigmoid(self._config["SigmoidSteepness"] * backup) * (1.0 - self._filter)
            printedNom, printedMax, printedMin = self._lithosim(mask)
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return l2Min, pvbMin, bestParams, bestMask


def parallel(): 
    SCALE = 4
    l2s = []
    pvbs = []
    epes = []
    shots = []
    targetsAll = []
    paramsAll = []
    cfg   = SimpleCfg("./config/simpleilt512.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho, multigpu=True)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    for idx in range(1, 11): 
        print(f"[SimpleILT]: Preparing testcase {idx}")
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        targetsAll.append(torch.unsqueeze(target, 0))
        paramsAll.append(torch.unsqueeze(params, 0))
    count = torch.cuda.device_count()
    print(f"Using {count} GPUs")
    while count > 0 and len(targetsAll) % count != 0: 
        targetsAll.append(targetsAll[-1])
        paramsAll.append(paramsAll[-1])
    print(f"Augmented to {len(targetsAll)} samples. ")
    targetsAll = torch.cat(targetsAll, 0)
    paramsAll = torch.cat(paramsAll, 0)

    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(targetsAll, paramsAll)
    runtime = time.time() - begin

    for idx in range(1, 11): 
        mask = bestMask[idx-1]
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb = test.run(mask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(mask, target, scale=SCALE)
        epe = epeIn + epeOut
        shot = shotCount.run(mask, shape=(512, 512))
        cv2.imwrite(f"./tmp/MOSAIC_test{idx}.png", (mask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.0f}; Shot {np.mean(shots):.0f}; SolveTime {runtime:.2f}s")


def serial(): 
    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    cfg   = SimpleCfg("./config/simpleilt2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    for idx in range(1, 11): 
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        
        begin = time.time()
        l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None)
        runtime = time.time() - begin
        
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb = test.run(bestMask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(bestMask, target, scale=SCALE)
        epe = epeIn + epeOut
        shot = shotCount.run(bestMask, shape=(512, 512))
        cv2.imwrite(f"./tmp/MOSAIC_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
        runtimes.append(runtime)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime {np.mean(runtimes):.2f}s")


if __name__ == "__main__": 
    serial()
    # parallel()


'''
fast backward (simple lithosim): 
[Testcase 1]: L2 43408; PVBand 52281; EPE 3; Shot: 723.0; SolveTime: 2.50s
[Testcase 2]: L2 35326; PVBand 41865; EPE 2; Shot: 623.0; SolveTime: 2.24s
[Testcase 3]: L2 75428; PVBand 78805; EPE 43; Shot: 873.0; SolveTime: 2.24s
[Testcase 4]: L2 13649; PVBand 22112; EPE 2; Shot: 781.0; SolveTime: 2.24s
[Testcase 5]: L2 37330; PVBand 54977; EPE 2; Shot: 604.0; SolveTime: 2.23s
[Testcase 6]: L2 35711; PVBand 51036; EPE 0; Shot: 659.0; SolveTime: 2.24s
[Testcase 7]: L2 29566; PVBand 44576; EPE 0; Shot: 555.0; SolveTime: 2.23s
[Testcase 8]: L2 14327; PVBand 20727; EPE 0; Shot: 876.0; SolveTime: 2.24s
[Testcase 9]: L2 45347; PVBand 64063; EPE 0; Shot: 617.0; SolveTime: 2.24s
[Testcase 10]: L2 8404; PVBand 16685; EPE 0; Shot: 809.0; SolveTime: 2.24s
[Result]: L2 33850; PVBand 44713; EPE 5.2; Shot 712.0; SolveTime 2.26s
'''

'''
exact backward (exact lithosim): 
[Testcase 1]: L2 41326; PVBand 51874; EPE 4; Shot: 715.0; SolveTime: 8.61s
[Testcase 2]: L2 31828; PVBand 41134; EPE 0; Shot: 595.0; SolveTime: 8.35s
[Testcase 3]: L2 71733; PVBand 81149; EPE 40; Shot: 781.0; SolveTime: 8.36s
[Testcase 4]: L2 12433; PVBand 23680; EPE 1; Shot: 714.0; SolveTime: 8.37s
[Testcase 5]: L2 34884; PVBand 56069; EPE 1; Shot: 558.0; SolveTime: 8.36s
[Testcase 6]: L2 34928; PVBand 50434; EPE 0; Shot: 591.0; SolveTime: 8.35s
[Testcase 7]: L2 26494; PVBand 44393; EPE 0; Shot: 480.0; SolveTime: 8.36s
[Testcase 8]: L2 12306; PVBand 20760; EPE 0; Shot: 782.0; SolveTime: 8.37s
[Testcase 9]: L2 39526; PVBand 65917; EPE 0; Shot: 644.0; SolveTime: 8.35s
[Testcase 10]: L2 8528; PVBand 16667; EPE 0; Shot: 647.0; SolveTime: 8.36s
[Result]: L2 31399; PVBand 45208; EPE 4.6; Shot 650.7; SolveTime 8.38s
'''

