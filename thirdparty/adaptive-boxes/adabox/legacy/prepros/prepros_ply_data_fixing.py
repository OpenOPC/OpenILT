import sys

from plyfile import PlyData
import pandas as pd
import numpy as np

# if len(sys.argv) < 3:
#     print("ERROR: args error: Needed: \n[1]in_path(with file.ply) \n[2]out_path (with file.npy at end)")
#     sys.exit()

in_path = "/Users/Juan/Desktop/Tesis Model/voxel/humboldt.ply"
out_path = "/Users/Juan/Desktop/Tesis Model/voxel/humboldt.npy"

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

np.save(out_path, data_to_save)
