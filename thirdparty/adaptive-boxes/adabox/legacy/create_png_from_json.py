
import sys

from adabox.plot_tools import plot_rectangles
from adabox.tools import load_from_json, Rectangle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
plt.ioff()

if len(sys.argv) < 3:
    print('ERROR args. Needed \n[1]in_path(.json) \n[2]out_path(.jpg)')
    sys.exit()

in_path = str(sys.argv[1])      # .json
out_path = str(sys.argv[2])     # .jpg

colors_list = list(colors._colors_full_map.values())
colors_len = len(colors_list)

json_data_raw = load_from_json(in_path)

json_data = np.array(json_data_raw['data'])
sep_value = float(json_data_raw['sep_value'])


recs = []
for jd in json_data:
    recs.append(Rectangle(jd[0], jd[1], jd[2], jd[3]))


plot_rectangles(recs, sep_value)
plt.savefig(out_path, dpi=3000)

