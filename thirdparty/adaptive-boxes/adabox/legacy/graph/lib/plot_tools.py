

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle

from adabox.legacy.graph.lib.PartitionRectangle import PartitionRectangle

cycle_color = cycle('bgrcmk')


def plot_rectangles(recs_arg):
    sep_value_arg = 1.0
    max_area_val = np.max([item.get_area() for item in recs_arg])

    fig = plt.figure()
    plt.axis('off')
    ax = fig.add_subplot(111)

    sep_to_plot = sep_value_arg / 2
    for rec_val in recs_arg:
        plot_rectangle(rec_val, sep_to_plot, max_area_val, ax)

    plt.axis('scaled')
    fig.tight_layout()


def plot_rectangle(rec_arg: PartitionRectangle, sep_to_plot_arg, max_area_arg, ax):
    p1 = np.array([rec_arg.x1 - sep_to_plot_arg, rec_arg.y1 - sep_to_plot_arg])
    p2 = np.array([rec_arg.x1 - sep_to_plot_arg, rec_arg.y2 + sep_to_plot_arg])
    p3 = np.array([rec_arg.x2 + sep_to_plot_arg, rec_arg.y1 - sep_to_plot_arg])
    p4 = np.array([rec_arg.x2 + sep_to_plot_arg, rec_arg.y2 + sep_to_plot_arg])

    color_tmp = rec_arg.partition_id

    ps = np.array([p1, p2, p4, p3, p1])

    max_n = 300
    max_log = np.log2(max_n + 1)
    area_ratio = (max_n*(rec_arg.get_area()/max_area_arg))+1
    line_w = np.log2(area_ratio)/max_log
    # plt.plot(ps[:, 0], ps[:, 1], linewidth=0.1*line_w + 0.08, c='black')
    plt.plot(ps[:, 0], ps[:, 1], linewidth=0.3*line_w + 0.1, c='black')

    # rect = matplotlib.patches.Rectangle((p1[0], p1[1]), p3[0] - p1[0], p2[1] - p1[1], color='black', lw=0)
    # rect = matplotlib.patches.Rectangle((p1[0], p1[1]), p3[0] - p1[0], p2[1] - p1[1], color=next(cycol), lw=0)
    rect = matplotlib.patches.Rectangle((p1[0], p1[1]), p3[0] - p1[0], p2[1] - p1[1], color=color_tmp, lw=0)
    ax.add_patch(rect)

