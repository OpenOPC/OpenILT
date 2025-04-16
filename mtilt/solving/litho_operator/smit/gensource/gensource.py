import sys
import os
from gwxopc_md import tcc
import gwxopc_md

SOURCEFILE = sys.argv[1]
assert isinstance(SOURCEFILE, str) and SOURCEFILE.split('.')[1] == 'src', "SOURCEFILE name error."
LOGLEVEL=2
# SOURCEFILE="/data/yangluo/OpenILT/lithosim_results/source/source.src"
sigmain=0.7
sigmaout=0.9
angle=30
rot=45
sourcetype="quasar" #dipole dipolex dipoley quasar quad annular
symmetry="D4" #D2 D4

gwxopc_md.initlog("./gensourcelog", LOGLEVEL)

tcc.setsrcparam(sigmain, sigmaout, angle, rot, sourcetype, symmetry, SOURCEFILE)
tcc.gensource()
gwxopc_md.dumpconfig("./gensource.yaml") #option
print("done.")
