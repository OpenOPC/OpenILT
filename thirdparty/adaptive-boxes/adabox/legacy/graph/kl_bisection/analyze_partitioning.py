
import numpy as np
import pandas as pd

# Partitions data
from lib.PartitionRectangle import PartitionRectangle
from lib.plot_tools import plot_rectangles

base_folder = "/Users/kolibri/PycharmProjects/gard/partitions/partitions_data/humboldt"

summary_groups_data_path = base_folder + '/' + 'summary_groups.csv'
x_units_path = base_folder + '/' + 'x_units.csv'
y_units_path = base_folder + '/' + 'y_units.csv'
group_details_path = base_folder + '/' + 'group_details.csv'


# Reading Data
summary_groups = pd.read_csv(summary_groups_data_path)

group_details = pd.read_csv(group_details_path)
x_units = pd.read_csv(x_units_path)
y_units = pd.read_csv(y_units_path)


# Partitioning data
partitioning_data_path = "/Users/Juan/django_projects/gard/partitions/kl_bisection/output/humboldt/humboldt_kl_partitions_2.npy"
partitioning_data = np.load(partitioning_data_path, allow_pickle=True)


# group_details_cpy = np.array(group_details)
group_details['group'] = -1
# Sorting data
n_group = 0
for p_data in partitioning_data:
    print("---------------------- " + "Group: " + str(n_group) + " ----------------------")
    group_data = list(p_data)
    # g_list = []
    for g_data in group_data:
        # print(g_data)
        [gi, gj] = g_data.split('_')
        condition = (group_details.gi == int(gi)) & (group_details.gj == int(gj))
        group_item = group_details[condition]
        p_index = group_item.index[0]
        group_details.loc[p_index, 'group'] = n_group
    n_group += 1

# Test if all was well
if sum(group_details.group == -1) != 0:
    print("ERROR: some partitions dont have a group")
else:
    print("Partitions are ok!")


# x interfaces
x_units['group'] = -1

for index, x_unit_row in x_units.iterrows():
    gi_l = x_unit_row['group_0']
    gj_l = x_unit_row['partition_0']

    gi_r = x_unit_row['group_1']
    gj_r = x_unit_row['partition_1']

    condition = (group_details.gi == gi_l) & (group_details.gj == gj_l)
    l_group = group_details[condition].group.iloc[0]

    condition = (group_details.gi == gi_r) & (group_details.gj == gj_r)
    r_group = group_details[condition].group.iloc[0]

    print("%d_%d = %d    %d_%d = %d" % (gi_l, gj_l, l_group, gi_r, gj_r, r_group))

    if l_group == r_group:
        x_unit_row.group = l_group
    else:
        x_unit_row.group = 2

# y interfaces
y_units['group'] = -1
for y_unit_row in y_units.itertuples(index=True):
    # print(y_unit_row)
    gi_l = y_unit_row.group_0
    gj_l = y_unit_row.partition_0

    gi_r = y_unit_row.group_1
    gj_r = y_unit_row.partition_1

    condition = (group_details.gi == gi_l) & (group_details.gj == gj_l)
    l_group = group_details[condition].group.iloc[0]

    condition = (group_details.gi == gi_r) & (group_details.gj == gj_r)
    r_group = group_details[condition].group.iloc[0]

    # print("%d_%d = %d    %d_%d = %d" % (gi_l, gj_l, l_group, gi_r, gj_r, r_group))

    if l_group == r_group:
        y_units.loc[y_unit_row.Index].group = l_group
    else:
        y_units.loc[y_unit_row.Index].group = 2


# Test if all was well X
groups_intfs_sum = sum(x_units.group == 0) + sum(x_units.group == 1) + sum(x_units.group == 2)
if (groups_intfs_sum - len(x_units)) != 0:
    print("ERROR: some interfaces dont have a group(x)")
else:
    print("Interfaces X are ok!")




# Test if all was well Y
groups_intfs_sum = sum(y_units.group == 0) + sum(y_units.group == 1) + sum(y_units.group == 2)
if (groups_intfs_sum - len(y_units)) != 0:
    print("ERROR: some interfaces dont have a group(Y)")
else:
    print("Interfaces Y are ok!")


# creating summary groups






# Creating Rectangles from Nodes (in this case: partition id matches with color code )
recs = []
for g_node_tmp in g.nodes.data(True):
    values = g_node_tmp[1]
    recs.append(
        PartitionRectangle(values['x1'],
                           values['x2'],
                           values['y1'],
                           values['y2'],
                           values['color']
                           )
    )

plot_rectangles(recs)


