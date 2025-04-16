#!/usr/bin/python3
import gwxopc_md as md
import numpy as np
print("**********test0**********")
t = md.DTensor([0.3, 0.2])
t.print()
print(t.to_numpy())
tnsr0 = md.DTensor([2,3,4]) # shape, zeros([shape]): double dtype
tnsr1 = md.DTensor([2,3,4])
tnsr2 = md.DTensor([2,3,4])
for i in range(tnsr0.shape()[0]):
    for j in range(tnsr0.shape()[1]):
        for k in range(tnsr1.shape()[2]):
            tnsr0.set([i,j,k], 100*i+j*10+k) # set value
print("tnsr0", tnsr0)
tnsr0.print()
tnsr1.setToVal(3) # all values == 3
print("tnsr1", tnsr1)
tnsr1.print()
md.tensorAdd(tnsr0, tnsr1, tnsr2) #tnsr2 = tnsr0 + tnsr1
print("tnsr2", tnsr2)
tnsr2.print()
print("**********test1**********")
tnsr = md.DTensor([3,4,5]) # shape
print("tnsr:\n", tnsr.to_numpy())
for i in range(tnsr.shape()[0]):
  tnsr_ref = tnsr.getRef([i,0,0],[1,4,5]) #first param: value, second: shape, return a view of DTensor.
  print("tnsr_ref before", i, ":\n", tnsr_ref.to_numpy())
  tnsr_ref.setToVal(i)
  print("tnsr_ref after", i, ":\n", tnsr_ref.to_numpy())
  print("tnsr:\n", tnsr.to_numpy()) # will change simultaneously: view

print("tnsr:\n", tnsr.to_numpy())


print("********test2**********")
numpy_int32 = np.array([2,3,4], dtype='int32')
print(numpy_int32.dtype)
tensor0 = md.ITensor(numpy_int32) #Itensor type
print("tensor0:", tensor0)
tensor0.print()

numpy_float32 = np.array([2,3,4], 'float32')
print(numpy_float32.dtype)
tensor2 = md.FTensor(numpy_float32)
print("tensor2:", tensor2)
tensor2.print()

numpy_float = np.array([2,3,4], 'float') # because default float dtype is 'float64'
print(numpy_float.dtype)
tensor3 = md.DTensor(numpy_float) # the pybind11 will convert it to double in tensor: 64bit
print("tensor3:", tensor3)
tensor3.print()

numpy_double = np.array([2,3,4], 'double')
print(numpy_double.dtype)
tensor4 = md.DTensor(numpy_double)
print("tensor4:", tensor4)
tensor4.print()

print("********test3**********")
numpy_a = np.array([1,2,3], "int32")
tensor_a_ref = md.ITensor(numpy_a) # return a view of np.ndarray
tensor_a_clone = md.ITensor(numpy_a, True) # not a view
numpy_a[0] = 10
tensor_a_ref.set([1], 20)
tensor_a_clone.set([2],30)
print(numpy_a) # 10, 20, 3
tensor_a_ref.print() #10, 20, 3
tensor_a_clone.print() # 1, 2, 30

tensor_b = md.ITensor([1,3]) # shape
numpy_b_ref = tensor_b.to_numpy() # return a 'np.ndarray' view of ITensor
numpy_b_clone = tensor_b.to_numpy(True)
tensor_b.set([0,0],10)
numpy_b_ref[0,1] = 20
numpy_b_clone[0,2] = 30
tensor_b.print() # 10, 20, 0
print(numpy_b_ref) # 10, 20, 0
print(numpy_b_clone) # 0, 0, 30


