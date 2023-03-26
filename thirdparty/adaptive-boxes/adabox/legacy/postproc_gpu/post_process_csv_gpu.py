
##
# Create simulation data from adaptive boxes results
##

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adabox.legacy.postproc_gpu.tools import create_groups, get_xy_units

colors_list = list(colors._colors_full_map.values())

in_path = "/adabox/decomposition/samples/decomposition_n_20.csv"  # .csv
out_path = "/adabox/decomposition/samples/postdata"  # without extension

data = np.array(pd.read_csv(in_path, header=None))
sep_value = 1           # it is a constant because adabox GPU returns partitions with this value. DONT CHANGE!

# data prepros adds boundaries to json rectangles
groups_details, summary = create_groups(data, sep_value)

for s in summary:
    print(s)
#
# # Plot Rectangles by groups
# plt.figure()
# for rec in groups_details:
#     x1 = rec[0]
#     x2 = rec[1]
#     y1 = rec[2]
#     y2 = rec[3]
#
#     p1 = np.array([x1, y1])
#     p2 = np.array([x1, y2])
#     p3 = np.array([x2, y1])
#     p4 = np.array([x2, y2])
#
#     ps = np.array([p1, p2, p4, p3, p1])
#     plt.plot(ps[:, 0], ps[:, 1])

# Save in a csv file
n_split_sep_value = 3
error_val = 0.6
y_units, x_units = get_xy_units(groups_details, sep_value, n_split_sep_value, error_val)

# Creating units
x_unit_list = []
for x_unit in x_units:
    # print(str(x_unit.group) + ' ' + str(x_unit.position))
    x_unit_list.append([x_unit.group[0][0],
                        x_unit.group[0][1],
                        x_unit.position[0],
                        x_unit.group[1][0],
                        x_unit.group[1][1],
                        x_unit.position[1],
                        ])

y_unit_list = []
for y_unit in y_units:
    # print(str(y_unit.group) + ' ' + str(y_unit.position))
    y_unit_list.append([y_unit.group[0][0],
                        y_unit.group[0][1],
                        y_unit.position[0],
                        y_unit.group[1][0],
                        y_unit.group[1][1],
                        y_unit.position[1],
                        ])

# x-units and y-units
# columns: (0: group, 1:partition, 2:interface_position) (3:group, 4:partition, 5:interface_position)
x_unit_df = pd.DataFrame(x_unit_list)
y_unit_df = pd.DataFrame(y_unit_list)

c_names_interfaces = ['group_0',
                      'partition_0',
                      'interface_position_0',
                      'group_1',
                      'partition_1',
                      'interface_position_1']

x_unit_df.columns = c_names_interfaces
y_unit_df.columns = c_names_interfaces

x_unit_df.to_csv(out_path + "/x_units.csv", header=True, index=None)
y_unit_df.to_csv(out_path + "/y_units.csv", header=True, index=None)

# Saving summary
summary_groups = pd.DataFrame(summary)
summary_groups.iloc[:, 2:] = summary_groups.iloc[:, 2:] * n_split_sep_value

summary_groups.columns = ['n_group', 'n_partitions', 'num_div_y', 'num_div_x']
summary_groups.to_csv(out_path + "/summary_groups.csv", header=True, index=None)

# saving groups details
groups_details_df = pd.DataFrame(groups_details)
# header: x1 x2 y1 y2 is_checked? gi gj
#     |-------------o(x2,y2)|
#     |                     |
#     |                     |
#     |o(x1,y1)-------------|
groups_details_subset_df = groups_details_df[[0, 1, 2, 3, 5, 6]]
groups_details_subset_df.columns = ['x1', 'x2', 'y1', 'y2', 'gi', 'gj']
# Normalizing
x_offset = abs(groups_details_subset_df.loc[:, 'x1'].min())
groups_details_subset_df.loc[:, 'x1'] = n_split_sep_value * (groups_details_subset_df.loc[:, 'x1'] + x_offset)
groups_details_subset_df.loc[:, 'x2'] = n_split_sep_value * (groups_details_subset_df.loc[:, 'x2'] + x_offset)

y_offset = abs(groups_details_subset_df.loc[:, 'y1'].min())
groups_details_subset_df.loc[:, 'y1'] = n_split_sep_value * (groups_details_subset_df.loc[:, 'y1'] + y_offset)
groups_details_subset_df.loc[:, 'y2'] = n_split_sep_value * (groups_details_subset_df.loc[:, 'y2'] + y_offset)

groups_details_subset_df.to_csv(out_path + "/group_details.csv", header=True, index=None)


# Plot Rectangles by groups
plt.figure()
for i in range(groups_details_subset_df.shape[0]):
    rec = groups_details_subset_df.iloc[i, :]
    x1 = rec[0]
    x2 = rec[1]
    y1 = rec[2]
    y2 = rec[3]

    p1 = np.array([x1, y1])
    p2 = np.array([x1, y2])
    p3 = np.array([x2, y1])
    p4 = np.array([x2, y2])

    ps = np.array([p1, p2, p4, p3, p1])
    plt.plot(ps[:, 0], ps[:, 1])

