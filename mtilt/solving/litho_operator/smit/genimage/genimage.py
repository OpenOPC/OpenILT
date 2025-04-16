import copy
import sys
import gwxopc_md
import numpy as np
from matplotlib import pyplot as plt


TCCSAVEPATH, layout = sys.argv[1], sys.argv[2]
assert isinstance(layout, str) and layout.split('.')[1] == 'gds', "layout file name error."
LOGLEVEL=2
gwxopc_md.initlog("./genimagelog", LOGLEVEL)

SIMULATION_CENTER=[34.5, 0] #nm unit
PIXELSIZE=7 #nm unit
IMAGEDIM=256 #pixel unit
# TCCSAVEPATH="/data/yangluo/OpenILT/lithosim_results/tcc_test_savepath"
FOCUS=18
IMAGE_Z=45

dose= 100.
TARGET_INTENSITY = 0.3
PHOTORESIST_SIGMOID_STEEPNESS = 50
MASKRELAX_SIGMOID_STEEPNESS = 4
TIME_STEP=1
OPC_ITERATION = 50
MASK_UPDATE_STEPSIZE =2
GAMMA = 2

NOMIMAL_DOSE=1
NOMIMAL_FOCUS=40
DEFOCUS_POSITIVE=100
DOSE_POSITIVE=1.03
DEFOCUS_NEGATIVE=100
DOSE_NEGETIVE=0.97
WEIGHT=2

# "/data/yangluo/OpenILT/gds_results/bar_1.gds"
gwxopc_md.setlayout("test_pattern", layout)
gwxopc_md.setlayer( "main", 0, 0, 0)
gwxopc_md.setlayer( "sraf", 1, 0, 0)
gwxopc_md.setmaintone(0.06, 180)
gwxopc_md.setfieldtone(0, 1)
gwxopc_md.setmaintone(1, 0)
gwxopc_md.setpixelsize(PIXELSIZE)
gwxopc_md.setimagedim(IMAGEDIM)
gwxopc_md.settcc(TCCSAVEPATH, FOCUS, IMAGE_Z)
gwxopc_md.setcenter(SIMULATION_CENTER)
gwxopc_md.dumpconfig("./config_image.yaml")

bitmap=gwxopc_md.computebitmap()
bitmap.runForward()
bitmapdata=bitmap.getNodeData("bitmapimage") # DTensor: Actually, the returned mask is empty with zero values, but that might be because the .gds file is corrupted.
bitmapdata_np=np.zeros(bitmapdata.to_numpy().shape)
bitmapdata_np[0,50:150, 50:100] = 1.
bitmapdata=gwxopc_md.DTensor(bitmapdata_np, True)
print(bitmapdata.to_numpy().shape)
plt.imshow(abs(bitmapdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)))
plt.show()

# bitmapdata[]
target=copy.deepcopy(bitmapdata_np).reshape(256 * 256)
cmigraph=gwxopc_md.computetrans(bitmapdata)
cmigraph.init()
cmigraph.runForward()
cmidata=cmigraph.getNodeData("cmiimage")
plt.imshow(abs(cmidata.to_numpy().reshape(IMAGEDIM, IMAGEDIM)))
plt.show()

aerialgraph=gwxopc_md.computeaerial(cmidata)
aerialgraph.init()
aerialgraph.runForward()
aerialdata=aerialgraph.getNodeData("aerialimage")
plt.imshow(aerialdata.to_numpy().reshape(IMAGEDIM, IMAGEDIM))
plt.show()

#################################
aerialdata = aerialgraph.getNodeData("aerialimage")
outai = aerialdata.to_numpy()
outai = outai.reshape(256 * 256)
npai = outai * dose


def Sigmoid(inputarray, steepness, target_intensity):
    '''
    The sigmoid function: [aerial image -> printed image]  or [continous mask -> 0-1 mask]
    '''
    return 1 / (1 + np.exp(-steepness * (inputarray - target_intensity)))

Z = Sigmoid(npai, steepness=PHOTORESIST_SIGMOID_STEEPNESS, target_intensity=TARGET_INTENSITY)
# print("max Z ={max},min Z={min}".format(max=Z.max(),min=Z.min()))

g = GAMMA * (Z - target)
# print("max Loss->Z ={max},min Loss->Z={min}".format(max=g.max(),min=g.min()))
gradin = GAMMA * PHOTORESIST_SIGMOID_STEEPNESS * Z * (1 - Z) * (Z - target)
# print("max printimage->ai ={max},min printimage->ai={min}".format(max=g2.max(),min=g2.min()))
gradin = gradin.reshape(1, 256, 256)
gradindata = gwxopc_md.DTensor(gradin)

aerialimage_grad = aerialgraph.getOutputGrad("aerialimage")
aerialimage_grad.setToVal(gradindata)
# aerialimage_grad.setToVal(1)
aerialgraph.runBackward()
aerialop_grad = aerialgraph.getInputGrad("aerialimage") # image-> cmiimage name error
# aerialop_grad.print()
cmiimage_grad = cmigraph.getOutputGrad("cmiimage")
# cmiimage_grad.print()

# cmiimage_grad.setToVal(aerialop_grad)
cmiimage_grad.setToVal(1)
cmigraph.runBackward()
gradin = cmigraph.getInputGrad("cmiimage")

# bitmap = cmigraph.getOutputGrad("bitmapimage")
# bitmap.setToVal(cmiop_grad)
# # cmi_grad.setToVal(1)
# cmigraph.runBackward()
# gradin = cmigraph.getInputGrad("bitmapimage")

gradin = gradin.to_numpy() #gradin[0,::]
# gradin.dtype = np.float32
gradin = gradin.reshape(256 * 256)


ai_gradin_list = [npai, gradin]
