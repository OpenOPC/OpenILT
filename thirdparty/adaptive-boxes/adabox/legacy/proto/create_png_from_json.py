
import matplotlib.pyplot as plt
from adabox.tools import *

# To save .png file use bellow code
# import matplotlib
# matplotlib.use('Agg')

file_name = 'house10'
path_base = '/Users/Juan/django_projects/py-ard/heuristic/results'

json_data = load_from_json(path_base + '/' + file_name + '.json')

data = np.array(json_data['data'])
sep_value = float(json_data['sep_value'])

# Plot Rectangles
plt.figure()
sep = sep_value/2
for rec in data:
    x1 = rec[0]
    x2 = rec[1]
    y1 = rec[2]
    y2 = rec[3]

    p1 = np.array([x1 - sep, y1 - sep])
    p2 = np.array([x1 - sep, y2 + sep])
    p3 = np.array([x2 + sep, y1 - sep])
    p4 = np.array([x2 + sep, y2 + sep])

    ps = np.array([p1, p2, p4, p3, p1])
    plt.plot(ps[:, 0], ps[:, 1], linewidth=0.9)

plt.savefig(path_base + '/' + file_name + '.png', dpi=600)
