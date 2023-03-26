
import matplotlib.pyplot as plt
import numpy as np


# convert from index xy to index ij
def xy_to_ij(x_arg, y_arg):
    index_x = x_arg
    index_j = int(np.round(abs(index_x - x_min) / sep_value))

    index_y = y_arg
    index_i = int(np.round(abs(y_max - index_y) / sep_value))

    return index_i, index_j


data = np.load("/Users/Juan/Desktop/Tesis Model/voxel/humboldt.npy")
out_path = "/Users/Juan/Desktop/Tesis Model/voxel/humboldt_binary_matrix.csv"

data = data

# plt.scatter(data[:, 0], data[:, 1], s=1)


data[:, 1] = -1*data[:, 1]





# Size data
m_data = data.shape[0]
n_data = data.shape[1]



# Get Sep Value
sep_value = 0

diffs = np.zeros(shape=[m_data, 1])

for i in range(len(data)-1):
    diffs[i] = abs(data[i+1, 0] - data[i, 0])

diffs_no_zero = diffs[diffs != 0]
sep_value = np.min(diffs_no_zero)


# Create Matrix
x_max = data[:, 0].max()
x_min = data[:, 0].min()

y_max = data[:, 1].max()
y_min = data[:, 1].min()

lx = x_max - x_min
ly = y_max - y_min

divs_i = int(np.round(ly/sep_value)) + 1
divs_j = int(np.round(lx/sep_value)) + 1

data_matrix = np.zeros(shape=[divs_i, divs_j])


for idx in range(len(data)):
    x_val = data[idx, 0]
    y_val = data[idx, 1]
    i_val, j_val = xy_to_ij(x_val, y_val)

    data_matrix[i_val, j_val] = int(1)


errors_vertical = list(np.where(data_matrix.sum(axis=0) == 0)[0])
errors_horizontal = list(np.where(data_matrix.sum(axis=1) == 0)[0])

data_matrix_fixed = data_matrix
for ev in errors_vertical:
    data_matrix_fixed = np.delete(data_matrix_fixed, ev, 1)

for eh in errors_horizontal:
    data_matrix_fixed = np.delete(data_matrix_fixed, eh, 0)


plt.subplot(111)
plt.imshow(data_matrix_fixed, cmap='Greys',  interpolation='nearest')


np.savetxt(out_path, data_matrix_fixed, delimiter=",", fmt='%d')



