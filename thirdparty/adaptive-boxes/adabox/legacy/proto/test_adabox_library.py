
import numpy as np
import matplotlib.pyplot as plt
from adabox import proc
from adabox.plot_tools import plot_rectangles, plot_rectangles_only_lines

# Input Path
in_path = '/Users/Juan/django_projects/adaptive-boxes/samples/sample_2.csv'

# Load Demo data with columns [x_position y_position flag]
data_2d = np.loadtxt(in_path, delimiter=",")

# Plot demo data
plt.scatter(data_2d[:, 0], data_2d[:, 1])
plt.axis('scaled')

# Decompose data in rectangles,
# returns a list of rectangles and a separation value needed to plot them
rectangles = []
(rectangles, sep_value) = proc.decompose(data_2d, 2)
print('Number of rectangles found: ' + str(len(rectangles)))

# Plot rectangles
plot_rectangles(rectangles, sep_value)

plot_rectangles_only_lines(rectangles, sep_value)
