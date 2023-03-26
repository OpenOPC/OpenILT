
from adabox import proc
from adabox.plot_tools import plot_rectangles, plot_rectangles_only_lines
import numpy as np
import matplotlib.pyplot as plt

# Input Path
in_path = './sample_data/humboldt_binary_matrix.csv'

# Load Demo data with columns [x_position y_position flag]
binary_matrix = np.loadtxt(in_path, delimiter=",")


# Plot demo data
plt.imshow(np.flip(binary_matrix, axis=0), cmap='magma',  interpolation='nearest')
