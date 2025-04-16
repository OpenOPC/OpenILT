import sys
from gwxopc_md import tcc
import gwxopc_md

TCCSAVEPATH, SOURCEFILE = sys.argv[1], sys.argv[2]

LOGLEVEL=2
# TCCSAVEPATH="/data/yangluo/OpenILT/lithosim_results/tcc_test_savepath"
# SOURCEFILE="/data/yangluo/OpenILT/lithosim_results/source/source.src"
SYMMETRY="D4" #D2 D4
POLTYPE="XY"

PIXELSIZE=7

gwxopc_md.initlog("/data/yangluo/OpenILT/smit/gentcc/gentcclog", LOGLEVEL)

tcc.setsource(SOURCEFILE, SYMMETRY, POLTYPE)
tcc.setNA(1.35)
tcc.setwavelength(193)
tcc.setfilm(1.66, 0.046, 90)
tcc.setfilm(1.59, 0.16, 36)
tcc.setfilm(1.44, 0.45, 125)
tcc.setsubstract(0.8831, 2.7779)
tcc.setcondition(18, 45)
tcc.setmedium(1.4366)
tcc.setgrid(10)
tcc.setredunction(4)
tcc.setpixelsize(PIXELSIZE)
tcc.setsavepath(TCCSAVEPATH)
######option######
tcc.setsavemode("ALL") #R P T ALL default T
tcc.setsaveorder(64) #default 64
######option######
tcc.computetcc()
gwxopc_md.dumpconfig("./tcc.yaml")
print("done.")
