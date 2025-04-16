#!/usr/bin/python3
import gwxopc_md as md
# print("*********************test0************************")
# g = md.loadGraphFromFile("../../calib/mask_graph.yml")
# g.init()
# g.runForward()
# g.runBackward()
# toneinfo = g.getNodeData("toneinfo")
# print(toneinfo)
# print(toneinfo.getValue())

print("*********************test1************************")

tnsr0 = md.DTensor([2,3,4])
tnsr1 = md.DTensor([2,3,4])
tnsr2 = md.DTensor([2,3,4])
for i in range(tnsr0.shape()[0]):
    for j in range(tnsr0.shape()[1]):
        for k in range(tnsr0.shape()[2]):
            tnsr0.set([i,j,k], 100*i+j*10+k)

for i in range(tnsr1.shape()[0]):
    for j in range(tnsr1.shape()[1]):
        for k in range(tnsr1.shape()[2]):
            tnsr1.set([i,j,k], 0.001*i+0.01*j + 0.1*k)
tnsr0.print()
tnsr1.print()
tnsr2.setToVal(0)

g = md.Graph()
g.addInputNode("tnsr0", tnsr0)
g.addInputNode("tnsr1", tnsr1)
g.addNode("add_op", "DTensorAddOp") # can not change any name: addNode(pre-defined name, operator)
g.addOutputNode("tnsr2")
g.link(["tnsr0", "tnsr1"], "add_op") # from [] -> to []
g.link(["add_op"], "tnsr2")
g.init()
g.runForward()
print("tnsr0:\n", tnsr0.to_numpy())
print("tnsr1:\n", tnsr1.to_numpy())
print("tnsr2:\n", g.getOutputData("tnsr2").to_numpy())
tnsr2.print() # not change

tnsr0_grad = g.getInputGrad("tnsr0") # return a 'DTensor' view of grad
tnsr1_grad = g.getInputGrad("tnsr1")
tnsr2_grad = g.getOutputGrad("tnsr2")
tnsr2_grad.setToVal(2)
g.runBackward()

print("tnsr0_grad:\n", tnsr0_grad.to_numpy())
print("tnsr1_grad:\n", tnsr1_grad.to_numpy())
print("tnsr2_grad:\n", tnsr2_grad.to_numpy())

g_str = md.dumpGraphToStr(g) # using strings to describe computational graph
print("\n\ndump graph:\n", md.dumpGraphToStr(g))
print("\n\ndump graph with data:\n", md.dumpGraphToStr(g,["tnsr0", "tnsr1"]))

print("*********************test2************************")
import numpy as np
from matplotlib import pyplot as plt
row_num = 256
col_num = 256
input_tnsr   = md.DTensor([1,row_num,col_num])
input_tnsr_ref = input_tnsr.centerRef([1,int(row_num/4),int(col_num/4)])
input_tnsr_ref.setToVal(1)

input_kernel = md.DTensor([3,3])
input_kernel.set([0,0], 0.05)
input_kernel.set([0,2], 0.05)
input_kernel.set([2,0], 0.05)
input_kernel.set([2,2], 0.05)

input_kernel.set([0,1], 0.15)
input_kernel.set([1,0], 0.15)
input_kernel.set([1,2], 0.15)
input_kernel.set([2,1], 0.15)

input_kernel.set([1,1], 0.20)

output_tnsr = md.DTensor([1,row_num,col_num])

g = md.Graph()
g.addInputNode("input0", input_tnsr)
g.addInputNode("input1", input_kernel)
g.addNode("multiple_node", "DTensorConv2DOp") # default: zero-padding
g.addOutputNode("output", output_tnsr)
g.link(["input0","input1"], "multiple_node")
g.link(["multiple_node"], "output")
g.init()
output_grad = g.getOutputGrad("output")
output_grad.setToVal(1)
g.runForward()
g.runBackward()
input0_grad = g.getInputGrad("input0")

# g_str = md.dumpGraphToStr(g) # using strings to describe computational graph
# print("\n\ndump graph:\n", md.dumpGraphToStr(g))
# print("\n\ndump graph with data:\n", md.dumpGraphToStr(g,["input0", "input1"]))

np.set_printoptions(threshold=np.inf)
plt.matshow(input_tnsr.to_numpy()[0,:,:])
plt.matshow(input_kernel.to_numpy())
plt.matshow(output_tnsr.to_numpy()[0,:,:])
plt.matshow(input0_grad.to_numpy()[0,:,:])
plt.show()
