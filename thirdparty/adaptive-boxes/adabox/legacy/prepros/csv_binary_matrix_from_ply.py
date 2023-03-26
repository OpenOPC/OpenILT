import sys

from plyfile import PlyData
import pandas as pd
import numpy as np


# convert from index xy to index ij
def xy_to_ij(x_arg, y_arg):
    index_x = x_arg
    index_j = int(np.round(abs(index_x - x_min) / sep_value))

    index_y = y_arg
    index_i = int(np.round(abs(y_max - index_y) / sep_value))

    return index_i, index_j


if len(sys.argv) < 3:
    print("ERROR: args error: Needed: \n[1]in_path(with file.ply) \n[2]out_path (with file.csv at end)")
    sys.exit()

in_path = str(sys.argv[1])
out_path = str(sys.argv[2])

print("Args summary: " + '\n In Path:' + in_path + '\n Out Path:' + out_path)

ply_data = PlyData.read(in_path)

# vertex
vertex = ply_data.elements[0].data
vertex_new = pd.DataFrame(vertex)
vertex_pos = vertex_new.loc[:, ['x', 'y', 'z']]

# plot_vertex_3d(vertex_pos)

# Handle z-data
z_min = vertex_pos.z.min()
z_max = vertex_pos.z.max()

# z-scale
z_scale = vertex_pos.z.drop_duplicates()
z_scale.reset_index(drop=True, inplace=True)
z_scale = np.sort(z_scale)

# Get Bottom
z_level = 0
vertex_bottom_set = vertex_pos[vertex_pos.z == z_scale[z_level]]

data_to_save = np.array(vertex_bottom_set)

# np.save(out_path, data_to_save)


# creating binary matrix
npy_data = data_to_save
npy_data[:, 1] = -1 * npy_data[:, 1]

# Size data
m_data = npy_data.shape[0]
n_data = npy_data.shape[1]

# Get Sep Value
sep_value = 0

diffs = np.zeros(shape=[m_data, 1])

for i in range(len(npy_data) - 1):
    diffs[i] = abs(npy_data[i + 1, 0] - npy_data[i, 0])

diffs_no_zero = diffs[diffs != 0]
sep_value = np.min(diffs_no_zero)

# Create Matrix
x_max = npy_data[:, 0].max()
x_min = npy_data[:, 0].min()

y_max = npy_data[:, 1].max()
y_min = npy_data[:, 1].min()

lx = x_max - x_min
ly = y_max - y_min

divs_i = int(np.round(ly / sep_value)) + 1
divs_j = int(np.round(lx / sep_value)) + 1

data_matrix = np.zeros(shape=[divs_i, divs_j])

for idx in range(len(npy_data)):
    x_val = npy_data[idx, 0]
    y_val = npy_data[idx, 1]
    i_val, j_val = xy_to_ij(x_val, y_val)

    data_matrix[i_val, j_val] = int(1)

data_m = divs_i
data_n = divs_j

text_file = open(out_path, "w")
# text_file.write('%d\n%d' % (data_m, data_n))

for i in range(data_m):
    for j in range(data_n):
        text_file.write('%d' % data_matrix[i][j])
        if j != (data_n - 1):
            text_file.write(',')
    text_file.write('\n')

text_file.close()

print("Work Finished!!")



