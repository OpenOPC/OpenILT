import os
import sys
sys.path.append(".")
import shutil
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func


from pycommon.settings import *
from gwxopc_md import tcc
import gwxopc_md


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


class _LithoSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, gwxopc_md, tcc_save_path, FOCUS, IMAGE_Z, SIMULATION_CENTER, PIXELSIZE, IMAGEDIM, NAME):


        bitmapdata_np = mask.unsqueeze(0).data.cpu().numpy() # [batchsize, IMAGEDIM, IMAGEDIM]
        bitmapdata = gwxopc_md.DTensor(bitmapdata_np, True)
        # print(bitmapdata.to_numpy().shape)
        # plt.imshow(abs(bitmapdata.to_numpy().reshape(IMAGEDIM,IMAGEDIM)))
        # plt.show()

        toneinfo = gwxopc_md.gettoneinfo()
        toneinfo = gwxopc_md.basictype.Str(toneinfo)
        cmigraph = gwxopc_md.Graph("trans")
        cmigraph.addInputNode("bitmap", bitmapdata)
        cmigraph.addInputNode("toneinfo", toneinfo)
        cmigraph.addNode("transop", "DMtransV0Op")
        cmigraph.addOutputNode("cmi")
        cmigraph.link(["bitmap", "toneinfo"], "transop")
        cmigraph.link(["transop"], "cmi")
        cmigraph.init()
        cmigraph.runForward()

        cmidata = cmigraph.getNodeData("cmi")
        # fig = plt.figure()
        # plt.title(f"0_iter_cmi")
        # im = plt.imshow(abs(cmidata.to_numpy().reshape(IMAGEDIM,IMAGEDIM)))
        # plt.colorbar(im, orientation='vertical')
        # plt.savefig(f"./lithosim_results/gds/gds_5_ilt_test/process/ilt/0_iter_cmi.png")

        tccsavepath = gwxopc_md.basictype.Str(tcc_save_path)
        focus = gwxopc_md.basictype.Double(FOCUS) # gwxopc_md.basictype.Float(focus).to_numpy()
        image_z = gwxopc_md.basictype.Double(IMAGE_Z)

        aerialgraph = gwxopc_md.Graph("aerial")
        aerialgraph.addInputNode("cmi", cmidata)
        aerialgraph.addInputNode("tccsavepath", tccsavepath)
        aerialgraph.addInputNode("focus", focus)
        aerialgraph.addInputNode("image_z", image_z)
        aerialgraph.addNode("aerialop", "DAerialV0Op")
        aerialgraph.addOutputNode("aerialimage")
        aerialgraph.link(["cmi", "tccsavepath", "focus", "image_z"], "aerialop")
        aerialgraph.link(["aerialop"], "aerialimage")
        aerialgraph.init()
        aerialgraph.runForward()

        aerialdata = aerialgraph.getNodeData("aerialimage")

        # fig = plt.figure()
        # im = plt.imshow(aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)) # (aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).max()
        # print((aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).max())
        # print((aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).min())
        # print((aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).mean())
        # np.savetxt('ai_ilt.csv', aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM), delimiter=',')
        # plt.colorbar(im, orientation='vertical')
        # plt.title("ai")
        # plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_1_ilt_test/process/ilt/0_iter_ai.png")
        # plt.show()

        ctx.saved = (bitmapdata, aerialgraph, cmigraph,  gwxopc_md, tcc_save_path, FOCUS, IMAGE_Z, SIMULATION_CENTER, PIXELSIZE, IMAGEDIM, NAME)

        return torch.tensor(aerialdata.to_numpy()[0]) # the target mask dtype==float32, if not the same, there should be an error when loss.backward()


    @staticmethod
    def backward(ctx, grad):
        (bitmapdata, aerialgraph, cmigraph, gwxopc_md, tcc_save_path, FOCUS, IMAGE_Z, SIMULATION_CENTER, PIXELSIZE, IMAGEDIM, NAME) = ctx.saved


        toneinfo = gwxopc_md.gettoneinfo()
        toneinfo = gwxopc_md.basictype.Str(toneinfo)
        cmigraph = gwxopc_md.Graph("trans")
        cmigraph.addInputNode("bitmap", bitmapdata)
        cmigraph.addInputNode("toneinfo", toneinfo)
        cmigraph.addNode("transop", "DMtransV0Op")
        cmigraph.addOutputNode("cmi")
        cmigraph.link(["bitmap", "toneinfo"], "transop")
        cmigraph.link(["transop"], "cmi")
        cmigraph.init()
        cmigraph.runForward()

        cmidata = cmigraph.getNodeData("cmi")
        # plt.imshow(abs(cmidata.to_numpy().reshape(IMAGEDIM,IMAGEDIM)))
        # plt.show()

        tccsavepath = gwxopc_md.basictype.Str(tcc_save_path)
        focus = gwxopc_md.basictype.Double(FOCUS)
        image_z = gwxopc_md.basictype.Double(IMAGE_Z)

        aerialgraph = gwxopc_md.Graph("aerial")
        aerialgraph.addInputNode("cmi", cmidata)
        aerialgraph.addInputNode("tccsavepath", tccsavepath)
        aerialgraph.addInputNode("focus", focus)
        aerialgraph.addInputNode("image_z", image_z)
        aerialgraph.addNode("aerialop", "DAerialV0Op")
        aerialgraph.addOutputNode("aerialimage")
        aerialgraph.link(["cmi", "tccsavepath", "focus", "image_z"], "aerialop")
        aerialgraph.link(["aerialop"], "aerialimage")
        aerialgraph.init()
        aerialgraph.runForward()

        grad = grad.reshape(1, IMAGEDIM, IMAGEDIM)
        gradindata = gwxopc_md.DTensor(grad.data.cpu().numpy())
        aerialimage_grad = aerialgraph.getOutputGrad("aerialimage")
        aerialimage_grad.setToVal(gradindata)  # gradindata.to_numpy()[0,::]
        aerialgraph.runBackward()
        aerialop_grad = aerialgraph.getInputGrad("cmi")

        cmi_grad = cmigraph.getOutputGrad("cmi")
        cmi_grad.setToVal(aerialop_grad)  # abs(aerialop_grad.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).max()
        cmigraph.runBackward()
        gradin = cmigraph.getInputGrad("bitmap")

        gradin = gradin.to_numpy()

        return torch.tensor(gradin)[0], None, None, None, None, None, None, None, None


class LithoSim(nn.Module):  # Mask -> Aerial -> Printed
    def __init__(self, TCCSAVEPATH, FOCUS, IMAGE_Z, SIMULATION_CENTER,
                 PIXELSIZE, IMAGEDIM, PRINTSTEEPNESS, DOSE, TARGET_THRESHOLD):
        super(LithoSim, self).__init__()

        self.SIMULATION_CENTER = SIMULATION_CENTER  # nm unit
        self.PIXELSIZE = PIXELSIZE  # nm unit
        self.IMAGEDIM = IMAGEDIM
        self.TCCSAVEPATH = TCCSAVEPATH
        self.FOCUS = FOCUS
        self.IMAGE_Z = IMAGE_Z
        self.DOSE = DOSE
        self.TARGET_THRESHOLD = TARGET_THRESHOLD
        self.PRINTSTEEPNESS = PRINTSTEEPNESS


    def forward(self, mask):
        ai= _LithoSim.apply(mask, gwxopc_md, self.TCCSAVEPATH, self.FOCUS, self.IMAGE_Z,
                                    self.SIMULATION_CENTER, self.PIXELSIZE,  self.IMAGEDIM, "norm")

        printedNom = torch.sigmoid(self.PRINTSTEEPNESS * (ai*self.DOSE - self.TARGET_THRESHOLD))
        # printedNom = torch.sigmoid(self.PRINTSTEEPNESS * (self.TARGET_INTENSITY - ai * self.DOSE))


        return ai, printedNom


class SimpleILT:
    def __init__(self,
                 IMAGEDIM=256,
                 lithosim=None,
                 device=DEVICE):
        super(SimpleILT, self).__init__()

        self.IMAGEDIM = IMAGEDIM
        self._device = device
        # Lithosim
        self._lithosim = lithosim

        self.stepsize = 0.5
        self.sigmoidsteepness = 4.0


    def solve(self, target, params,
              iteration=20,
              dir='./',
              index=1,):

        # Initialize
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)


        # backup = params
        params = params.clone().detach().requires_grad_(True)
        target = target

        # Optimizer
        opt = optim.SGD([params], lr=self.stepsize)
        # opt = optim.Adam([params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        l2losses = []

        dir = f"{dir}/process/ilt"
        if not os.path.exists(dir):
            os.makedirs(dir)

        for idx in range(iteration):

            mask = torch.sigmoid(self.sigmoidsteepness * params)

            # show_mask = mask.data.cpu().numpy()
            # fig = plt.figure()
            # plt.imshow(show_mask, cmap="gray")
            # plt.show()

            ai, printedNom = self._lithosim(mask)

            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            l2losses.append(l2loss.item())
            loss = l2loss

            print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; ")


            if bestParams is None or bestMask is None or loss.item() < lossMin:

                lossMin, l2Min = loss.item(), l2loss.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()  # (mask * 255).data.cpu().numpy()

            opt.zero_grad()
            loss.backward() # not leaf node

            grad = params.grad.data
            # torch.cuda.empty_cache()
            # torch.autograd.backward(loss, retain_graph=False)

            fig = plt.figure()
            plt.title(f"target")
            im=plt.imshow(target.data.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM))
            plt.colorbar(im, orientation='vertical')
            plt.savefig(f"{dir}/target.png")

            fig = plt.figure()
            plt.title(f"{idx}_iter_ai")
            im=plt.imshow(ai.data.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM))
            plt.colorbar(im, orientation='vertical')
            plt.savefig(f"{dir}/{idx}_iter_ai.png")

            # fig = plt.figure()
            # plt.title(f"{idx}_iter_cmi")
            # im=plt.imshow(cmi.data.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM), cmap="gray")
            # plt.colorbar(im, orientation='vertical')
            # plt.savefig(f"{dir}/{idx}_iter_cmi.png")

            fig = plt.figure()
            plt.title(f"{idx}_iter_printmask")
            im=plt.imshow(printedNom.data.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM))
            plt.colorbar(im, orientation='vertical')
            plt.savefig(f"{dir}/{idx}_iter_printmask.png")

            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= 0.5] = 1

            fig = plt.figure()
            plt.title(f"{idx}_iter_printBimask")
            im=plt.imshow(binaryNom.data.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM))
            plt.colorbar(im, orientation='vertical')
            plt.savefig(f"{dir}/{idx}_iter_printBimask.png")

            fig = plt.figure()
            plt.title(f"{idx}_iter_mask")
            im=plt.imshow(mask.data.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM))
            plt.colorbar(im, orientation='vertical')
            plt.savefig(f"{dir}/{idx}_iter_mask.png")

            fig = plt.figure()
            # np.savetxt('gradilt.csv', grad.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM), delimiter=',')
            plt.title(f"{idx}_iter_grad")
            im=plt.imshow(grad.numpy().reshape(self.IMAGEDIM, self.IMAGEDIM))
            plt.colorbar(im, orientation='vertical')
            plt.savefig(f"{dir}/{idx}_iter_grad.png")

            opt.step()


        if True:
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.plot(list(range(iteration)), l2losses, c='red', label='l2loss')
            ax.legend(loc=0, fontsize=20)
            ax.grid()
            plt.xticks(np.arange(0, iteration + 1, 3))
            plt.savefig(f'{dir}/loss.png')

        return l2Min, pvbMin, bestParams, bestMask


def gensource(SOURCEFILE):
    sigmain = 0.7
    sigmaout = 0.9
    angle = 30
    rot = 45
    sourcetype = "quasar"  # dipole dipolex dipoley quasar quad annular
    symmetry = "D4"  # D2 D4

    gwxopc_md.initlog("./gensourcelog", 2)

    tcc.setsrcparam(sigmain, sigmaout, angle, rot, sourcetype, symmetry, SOURCEFILE)
    tcc.gensource()

def gentcc(SOURCEFILE, TCCSAVEPATH, PIXELSIZE, FOCUS=18, IMAGE_Z=45):
    SYMMETRY = "D4"
    POLTYPE = "XY"
    gwxopc_md.initlog("./smit/gentcc/gentcclog", 2)

    tcc.setsource(SOURCEFILE, SYMMETRY, POLTYPE)
    tcc.setNA(1.35)
    tcc.setwavelength(193)
    tcc.setfilm(1.66, 0.046, 90)
    tcc.setfilm(1.59, 0.16, 36)
    tcc.setfilm(1.44, 0.45, 125)
    tcc.setsubstract(0.8831, 2.7779)
    tcc.setcondition(FOCUS, IMAGE_Z)
    tcc.setmedium(1.4366)
    tcc.setgrid(10)
    tcc.setredunction(4)
    tcc.setpixelsize(PIXELSIZE)
    tcc.setsavepath(TCCSAVEPATH)
    ######option######
    tcc.setsavemode("ALL")  # R P T ALL default T
    tcc.setsaveorder(64)  # default 64
    ######option######
    tcc.computetcc()
    gwxopc_md.dumpconfig("./tcc.yaml")

def gwx_init(gds_filename, SOURCEFILE, PIXELSIZE, TCCSAVEPATH, FOCUS, IMAGE_Z,
             SIMULATION_CENTER, IMAGEDIM, PRE_TARGET_INTENSITY, savedir):


    gensource(SOURCEFILE)
    gentcc(SOURCEFILE, TCCSAVEPATH, PIXELSIZE, FOCUS, IMAGE_Z)

    gwxopc_md.setlayout("test_pattern", gds_filename)
    gwxopc_md.setlayer("main", 0, 0, 0)
    gwxopc_md.setlayer("sraf", 1, 0, 0)
    gwxopc_md.setmaintone(0.06, 180)
    gwxopc_md.setfieldtone(0, 1)
    gwxopc_md.setmaintone(1, 0)
    gwxopc_md.setpixelsize(PIXELSIZE)
    gwxopc_md.setimagedim(IMAGEDIM)
    gwxopc_md.settcc(TCCSAVEPATH, FOCUS, IMAGE_Z)
    gwxopc_md.setcenter(SIMULATION_CENTER)
    gwxopc_md.dumpconfig("./config_genimage_split.yaml")

    bitmap = gwxopc_md.computebitmap()
    bitmap.runForward()
    bitmapdata = bitmap.getNodeData("bitmapimage")
    print(bitmapdata.to_numpy().shape)
    im = plt.imshow(abs(bitmapdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)))
    plt.colorbar(im, orientation='vertical')
    plt.title("bitmapdata")
    plt.savefig(f"{savedir}/bitmapdata.png")
    # plt.show()

    target = np.zeros_like(bitmapdata.to_numpy())
    target[bitmapdata.to_numpy() > PRE_TARGET_INTENSITY] = 1
    fig = plt.figure()
    im = plt.imshow(target[0])
    plt.colorbar(im, orientation='vertical')
    plt.title("target bi mask")
    plt.savefig(f"{savedir}/target.png")
    # plt.show()

    return target[0], target[0] * 2.0 - 1.0


def gwx(gds_index, SOURCEFILE, PIXELSIZE, TCCSAVEPATH, FOCUS, IMAGE_Z,
        SIMULATION_CENTER, IMAGEDIM, TARGET_THRESHOLD, PRE_TARGET_INTENSITY, PRINTSTEEPNESS, DOSE, iteration=60):


    gds_filename = f"./lithosim_results/gds/bar_{gds_index}.gds"
    save_dir = f"./lithosim_results/gds/gds_{gds_index}_ilt_test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    litho = LithoSim(TCCSAVEPATH, FOCUS, IMAGE_Z, SIMULATION_CENTER, PIXELSIZE, IMAGEDIM, PRINTSTEEPNESS, DOSE, TARGET_THRESHOLD)
    solver = SimpleILT(IMAGEDIM, litho)
    target, params = gwx_init(gds_filename, SOURCEFILE, PIXELSIZE,
                              TCCSAVEPATH, FOCUS, IMAGE_Z,
                              SIMULATION_CENTER, IMAGEDIM, PRE_TARGET_INTENSITY,
                              save_dir)

    solver.solve(target, params, iteration=iteration, dir=save_dir, index=gds_index)


if __name__ == "__main__":

    gds_index = 3
    SOURCEFILE = "./lithosim_results/source/source.src"
    PIXELSZE = 14
    IMAGE_Z = 45
    FOCUS = 18
    savepath = "./lithosim_results/tcc_save_path/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    TCCSAVEPATH = f"{savepath}/{FOCUS}_{IMAGE_Z}"
    SIMULATION_CENTER = [200, 250]
    IMAGEDIM = 256
    TARGET_THRESHOLD = 0.146
    PRINTSTEEPNESS = 50.
    DOSE=1.
    PRE_TARGET_INTENSITY=0.7


    gwx(gds_index=gds_index,
        SOURCEFILE=SOURCEFILE,
        PIXELSIZE=PIXELSZE,
        TCCSAVEPATH=TCCSAVEPATH,
        FOCUS=FOCUS, IMAGE_Z=IMAGE_Z,
        SIMULATION_CENTER=SIMULATION_CENTER,
        IMAGEDIM=IMAGEDIM,
        TARGET_THRESHOLD=TARGET_THRESHOLD,
        PRINTSTEEPNESS=PRINTSTEEPNESS,
        PRE_TARGET_INTENSITY=PRE_TARGET_INTENSITY,
        DOSE=DOSE
        )
