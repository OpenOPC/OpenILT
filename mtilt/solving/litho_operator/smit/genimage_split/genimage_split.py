import os
import gwxopc_md
import numpy as np
from matplotlib import pyplot as plt
import torch


gds_num = 5
save_dir = f"./lithosim_results/gds/gds_{gds_num}_gts"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
LOGLEVEL=2
gwxopc_md.initlog("./genimagelog", LOGLEVEL)

SIMULATION_CENTER=[360, 250] #nm unit
PIXELSIZE=14 #nm unit
IMAGEDIM=256 #pixel unit

TCCSAVEPATH="./lithosim_results/pixelsize14/tcc_savepath_18_45"
FOCUS=18
IMAGE_Z=45

dose= 1.
PRE_TARGET_INTENSITY = 0.7
TARGET_THRESHOLD = 0.146
PHOTORESIST_SIGMOID_STEEPNESS = 50
MASKRELAX_SIGMOID_STEEPNESS = 4
TIME_STEP=1
OPC_ITERATION = 50
MASK_UPDATE_STEPSIZE=2
GAMMA = 2


###################
gwxopc_md.setlayout("test_pattern", f"/data/yangluo/OpenILT/lithosim_results/gds/bar_{gds_num}.gds")
gwxopc_md.setlayer( "main", 0, 0, 0)
gwxopc_md.setlayer( "sraf", 1, 0, 0)
gwxopc_md.setmaintone(0.06, 180)
gwxopc_md.setfieldtone(0, 1)
gwxopc_md.setmaintone(1, 0)
gwxopc_md.setpixelsize(PIXELSIZE)
gwxopc_md.setimagedim(IMAGEDIM)
gwxopc_md.settcc(TCCSAVEPATH, FOCUS, IMAGE_Z)
gwxopc_md.setcenter(SIMULATION_CENTER)
gwxopc_md.dumpconfig("./config_genimage_split.yaml")

bitmap=gwxopc_md.computebitmap()
bitmap.runForward()
bitmapdata=bitmap.getNodeData("bitmapimage")

print(bitmapdata.to_numpy().shape)
im=plt.imshow(abs(bitmapdata.to_numpy().reshape(IMAGEDIM,IMAGEDIM)))
plt.colorbar(im,orientation='vertical')
plt.title("bitmapdata")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/bitmapdata.png")
plt.show()




target = np.zeros_like(bitmapdata.to_numpy())
target[bitmapdata.to_numpy()>PRE_TARGET_INTENSITY] = 1
target[bitmapdata.to_numpy()<=PRE_TARGET_INTENSITY] = 0
im=plt.imshow(target[0])
plt.colorbar(im,orientation='vertical')
plt.title("target bi mask")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/target.png")
plt.show()

P = target*2.0 - 1.0
mask = torch.sigmoid(torch.tensor(P)*4.0).numpy()
im=plt.imshow(mask[0])
plt.colorbar(im,orientation='vertical')
plt.title("input sigmoid mask")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/inputsigmoidmask.png")
plt.show()

# mask = gwxopc_md.DTensor(target, True)
mask = gwxopc_md.DTensor(mask, True)
target = target.reshape(IMAGEDIM*IMAGEDIM)




toneinfo=gwxopc_md.gettoneinfo()
toneinfo=gwxopc_md.basictype.Str(toneinfo)
cmigraph=gwxopc_md.Graph("trans")
cmigraph.addInputNode("bitmap", mask)
cmigraph.addInputNode("toneinfo", toneinfo)
cmigraph.addNode("transop", "DMtransV0Op")
cmigraph.addOutputNode("cmi")
cmigraph.link(["bitmap", "toneinfo"], "transop")
cmigraph.link(["transop"], "cmi")
cmigraph.init()
cmigraph.runForward()

cmidata=cmigraph.getNodeData("cmi")

im=plt.imshow(abs(cmidata.to_numpy().reshape(IMAGEDIM,IMAGEDIM)))
plt.colorbar(im,orientation='vertical')
plt.title("cmidata")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/cmi.png")
plt.show()

tccsavepath=gwxopc_md.basictype.Str(TCCSAVEPATH)
focus=gwxopc_md.basictype.Double(FOCUS)
image_z=gwxopc_md.basictype.Double(IMAGE_Z)

aerialgraph=gwxopc_md.Graph("aerial")
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

aerialdata=aerialgraph.getNodeData("aerialimage")
# print((aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).max())
# print((aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).min())
# print((aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)).mean())
# np.savetxt('ai.csv', aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM), delimiter=',')
im=plt.imshow(aerialdata.to_numpy().reshape(IMAGEDIM,IMAGEDIM))
plt.colorbar(im,orientation='vertical')
plt.title("ai")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/aerial.png")
plt.show()

#############################
outai = aerialdata.to_numpy()
outai = outai.reshape(IMAGEDIM * IMAGEDIM)
npai = outai * dose


def Sigmoid(inputarray, steepness, target_intensity):
    '''
    The sigmoid function: [aerial image -> printed image]  or [continous mask -> 0-1 mask]
    '''
    return 1 / (1 + np.exp(-steepness * (inputarray - target_intensity)))

Z = Sigmoid(npai, steepness=PHOTORESIST_SIGMOID_STEEPNESS, target_intensity=TARGET_THRESHOLD)
# print("max Z ={max},min Z={min}".format(max=Z.max(),min=Z.min()))
im=plt.imshow(Z.reshape(IMAGEDIM,IMAGEDIM))
plt.colorbar(im,orientation='vertical')
plt.title("Z")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/Z.png")
plt.show()



g = GAMMA * (Z - target)
# self.utils.ShowHeatmap(target=g,length=self.length,name=self.top,i=i,type="Loss->Z")
# print("max Loss->Z ={max},min Loss->Z={min}".format(max=g.max(),min=g.min()))
gradin = GAMMA * PHOTORESIST_SIGMOID_STEEPNESS * Z * (1 - Z) * (Z - target)
gradin = gradin.reshape(1, IMAGEDIM, IMAGEDIM)

im=plt.imshow(gradin.reshape(IMAGEDIM,IMAGEDIM))
plt.colorbar(im,orientation='vertical')
plt.title("gradin")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/gradin.png")
plt.show()

gradindata = gwxopc_md.DTensor(gradin)

aerialimage_grad = aerialgraph.getOutputGrad("aerialimage")
aerialimage_grad.setToVal(gradindata)
aerialgraph.runBackward()
aerialop_grad = aerialgraph.getInputGrad("cmi")

im=plt.imshow(abs(aerialop_grad.to_numpy().reshape(IMAGEDIM,IMAGEDIM)))
plt.colorbar(im,orientation='vertical')
plt.title("cmi grad map")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/cmi_grad.png")
plt.show()

cmi_grad = cmigraph.getOutputGrad("cmi")
cmi_grad.setToVal(aerialop_grad)
cmigraph.runBackward()
gradmask = cmigraph.getInputGrad("bitmap")


im=plt.imshow(gradmask.to_numpy().reshape(IMAGEDIM,IMAGEDIM))
plt.colorbar(im,orientation='vertical')
plt.title("input mask grad map")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/grad.png")
plt.show()


def sigmoid(data):
    return 1/ (1+np.exp(-data))

gradlayout = gradmask.to_numpy().reshape(IMAGEDIM,IMAGEDIM) * sigmoid(P[0]*4.0) * (1-sigmoid(P[0]*4.0)) * 4
# np.savetxt('graddemo.csv', gradlayout, delimiter=',')

im=plt.imshow(gradlayout.reshape(IMAGEDIM,IMAGEDIM))
plt.colorbar(im,orientation='vertical')
plt.title("params grad map")
plt.savefig(f"/data/yangluo/OpenILT/lithosim_results/gds/gds_{gds_num}_gts/layout_grad.png")
plt.show()


ai_gradin_list = [npai, gradin]
