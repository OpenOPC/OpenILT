
import numpy as np
import pandas as pd


from adabox.tools import create_2d_data_from_vertex

in_path = '/Users/Juan/django_projects/adaptive-boxes/prepros/data_npy/complex11.npy'
out_path = '/Users/Juan/django_projects/adaptive-boxes/samples/sample_2.csv'


np_data = np.load(in_path)
vertex_bottom_set = pd.DataFrame(np_data)
vertex_bottom_set.columns = ['x', 'y', 'z']
# Create Global Matrix of points: []
data_2d = create_2d_data_from_vertex(vertex_bottom_set)

np.savetxt(out_path, data_2d, delimiter=",", fmt='%f')
