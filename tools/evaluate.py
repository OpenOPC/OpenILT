import os.path
import numpy as np

from pycommon.settings import *
import pylitho.simple as lithosim

import pyilt.evaluation as evaluation



def evalue(mask_saved_dirpath):
    SCALE = 1
    l2s = []
    ious = []
    pvbs = []
    pvb_news = []
    epes = []
    shots = []
    runtimes = []

    litho = lithosim.LithoSim("./config/lithosimple.txt")


    # 4. load data
    for idx in range(1, 11):
        dir = f"{mask_saved_dirpath}/pkl"
        bestMask = torch.load(f"{dir}/{idx - 1}_bestmask.pkl", map_location="cuda:0")
        target = torch.load(f"{dir}/{idx - 1}_target.pkl", map_location="cuda:0")
        l2, pvb, epe, shot, iou, pvb_new = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=False)
        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")


        l2s.append(l2)
        ious.append(iou)
        pvbs.append(pvb)
        pvb_news.append(pvb_new)
        epes.append(epe)
        shots.append(shot)
        # runtimes.append(runtime)

    print(
        f"[Result]: L2 {np.mean(l2s):.0f}; IOU {np.mean(ious):.4f}; PVBand {np.mean(pvbs):.0f}; "
        f"PVBand_new {np.mean(pvb_news):.4f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}")


if __name__ == "__main__":
    mask_saved_dirpath = ''
    assert os.path.exists(mask_saved_dirpath), "can not find the saved masks"
    evalue(mask_saved_dirpath)

