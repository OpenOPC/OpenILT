
import ast
import os
import shutil
import numpy as np

import matplotlib.colors as colors
import networkx as nx
import pandas as pd

colors_list = list(colors._colors_full_map.values())


def create_folder(path_tmp_arg):
    if not os.path.isdir(path_tmp_arg):
        os.makedirs(path_tmp_arg)


def delete_folder(path_tmp_arg):
    try:
        shutil.rmtree(path_tmp_arg)
    except:
        print("Error deleting %s" % path_tmp_arg)


# Init - Load GEXF File
data_path = "/Users/Juan/django_projects/gard/partitions/kl_bisection/output/humboldt/humboldt_kl_partitions_2.gexf"
model_name = "hum_10_3"
# SCALE: IMPORTANT MUST BE INT
SCALE = 10/3

# read data
G = nx.read_gexf(data_path)

# Partitions = Nodes
nodes_data = []
for g_node in G.nodes.data():
    # print(g_node[1])
    nodes_data.append(g_node[1])

partitions_df = pd.DataFrame(nodes_data)
partitions_df[['x1', 'x2', 'y1', 'y2', 'x', 'y']] = SCALE * partitions_df[['x1', 'x2', 'y1', 'y2', 'x', 'y']]
# partitions_df[['x1', 'x2', 'y1', 'y2', 'x', 'y']] = int(SCALE) * partitions_df[['x1', 'x2', 'y1', 'y2', 'x', 'y']]
partitions_df[['area']] = (SCALE * SCALE) * partitions_df[['area']]
# partitions_df[['area']] = int(SCALE * SCALE) * partitions_df[['area']]

partitions_df['dim_x'] = np.round(abs(partitions_df['x1'] - partitions_df['x2']))
partitions_df['dim_y'] = np.round(abs(partitions_df['y1'] - partitions_df['y2']))
partitions_df['dim_x_y_code'] = partitions_df['dim_x'].astype(str) + "_" + partitions_df['dim_y'].astype(str)
partitions_df['device_group'] = list(map(lambda x: colors_list.index(x), partitions_df.color))

partitions_df['gi'] = -1
partitions_df['gj'] = -1

# sort by area
partitions_df = partitions_df.sort_values(by="area")

pg_list = []
device_groups = partitions_df.groupby('device_group')
for idx, dg in device_groups:
    # print(dg)
    print(idx)

    processing_groups = dg.groupby('dim_x_y_code')
    # pg_list = []
    pg_counter = 0
    for idx2, pg in processing_groups:
        # print(pg)
        partitions_df.loc[pg.index, 'gi'] = pg_counter
        partitions_df.loc[pg.index, 'gj'] = list(range(0, pg.shape[0]))
        pg_list.append(
            (pg_counter, pg.shape[0], pg.dim_y.iloc[0], pg.dim_x.iloc[0], idx)
        )
        pg_counter += 1

# Summary group; n_group  n_partitions  num_div_y  num_div_x device_group labels
summary_groups = pd.DataFrame(pg_list)
summary_groups.columns = list(['n_group', 'n_partitions', 'num_div_y', 'num_div_x', 'device_group'])

# Nodes Partitions Summary
partitions_df_desc = partitions_df.groupby('device_group').describe()
# Memory required
total_area = sum(partitions_df.area)
# PartitionGroupSize = (n_partitions * 6 * m * n * scale)
# + (3 * m * n * scale) + (n*scale)^2 + (m*scale)^2 + (m * n * scale) = ~12 times
elements_required = 12 * total_area
size_bytes = elements_required * 8  # double
size_mb = size_bytes / (1024 ** 2)
print('Processing groups size: ' + str(size_mb) + ' Mb')
print('Processing groups size: ' + str(size_mb / 1024) + ' Gb')
# partitions_df.groupby('device_group').get_group(1).area.sum()
# Interfaces - Edges
# ['group_0', 'partition_0', 'interface_position_0', 'group_1', 'partition_1', 'interface_position_1']
edges_data = []
intf_data = []
for e in G.edges.data():
    # print(e)
    edges_data.append(e)

    condition = partitions_df.label == G.nodes.get(e[0])['label']
    n_0_info = partitions_df[condition]

    g_index = partitions_df[condition].color.iloc[0]
    n_0_g = colors_list.index(g_index)

    condition = partitions_df.label == G.nodes.get(e[1])['label']
    n_1_info = partitions_df[condition]

    g_index = partitions_df[condition].color.iloc[0]
    n_1_g = colors_list.index(g_index)

    # print('%d %d' % (n_0_g, n_1_g))

    group_of_pair = n_0_g
    if n_0_g != n_1_g:
        group_of_pair = -1

    edge_attr_dict = ast.literal_eval(e[2]['attr_dict'])
    e_type = edge_attr_dict['color']  # color is the type of the edge: 1 for y-aligned 0 for x-aligned

    device_group = group_of_pair

    if e_type == 1:  # y-unit
        # print('x-axis shared')
        intf_coords = [max([n_0_info.x1.iloc[0], n_1_info.x1.iloc[0]]), min([n_0_info.x2.iloc[0], n_1_info.x2.iloc[0]])]
        intf_num = abs(intf_coords[1] - intf_coords[0])

        if n_1_info.y2.iloc[0] > n_0_info.y1.iloc[0]:
            group_0 = n_1_info.gi.iloc[0]
            partition_0 = n_1_info.gj.iloc[0]
            interface_position_0 = intf_coords[0] - n_1_info.x1.iloc[0]
            # interface_position_0 = intf_coords[0] - n_1_info.x1.iloc[0]
            device_id_0 = n_1_info.device_group.iloc[0]

            group_1 = n_0_info.gi.iloc[0]
            partition_1 = n_0_info.gj.iloc[0]
            interface_position_1 = intf_coords[0] - n_0_info.x1.iloc[0]
            # interface_position_1 = intf_coords[0] - n_0_info.x1.iloc[0]
            device_id_1 = n_0_info.device_group.iloc[0]

        else:
            group_0 = n_0_info.gi.iloc[0]
            partition_0 = n_0_info.gj.iloc[0]
            interface_position_0 = intf_coords[0] - n_0_info.x1.iloc[0]
            # interface_position_0 = intf_coords[0] - n_0_info.x1.iloc[0]
            device_id_0 = n_0_info.device_group.iloc[0]

            group_1 = n_1_info.gi.iloc[0]
            partition_1 = n_1_info.gj.iloc[0]
            interface_position_1 = intf_coords[0] - n_1_info.x1.iloc[0]
            # interface_position_1 = intf_coords[0] - n_1_info.x1.iloc[0]
            device_id_1 = n_1_info.device_group.iloc[0]

        # ['group_0', 'partition_0', 'interface_position_0', 'group_1', 'partition_1', 'interface_position_1']
        # unit_type = 'y-unit'
        unit_type = 1   # 'y-unit'

        # for offset in range(0, int(intf_num)):
        #     interface_position_0 += 1
        #     interface_position_1 += 1
        #     intf_data.append((group_0, partition_0, interface_position_0, group_1, partition_1, interface_position_1,
        #                       unit_type, device_group, device_id_0, device_id_1))

        intf_data.append((group_0, partition_0, interface_position_0, group_1, partition_1, interface_position_1,
                                                unit_type, device_group, device_id_0, device_id_1, intf_num))

    else:  # x-unit
        # print('y-axis shared')
        intf_coords = [min([n_0_info.y2.iloc[0], n_1_info.y2.iloc[0]]), max([n_0_info.y1.iloc[0], n_1_info.y1.iloc[0]])]
        intf_num = abs(intf_coords[1] - intf_coords[0])

        if n_1_info.x2.iloc[0] > n_0_info.x1.iloc[0]:
            group_0 = n_0_info.gi.iloc[0]
            partition_0 = n_0_info.gj.iloc[0]
            interface_position_0 = n_0_info.y2.iloc[0] - intf_coords[0]
            # interface_position_0 = n_0_info.y2.iloc[0] - intf_coords[0]
            device_id_0 = n_0_info.device_group.iloc[0]

            group_1 = n_1_info.gi.iloc[0]
            partition_1 = n_1_info.gj.iloc[0]
            interface_position_1 = n_1_info.y2.iloc[0] - intf_coords[0]
            # interface_position_1 = n_1_info.y2.iloc[0] - intf_coords[0]
            device_id_1 = n_1_info.device_group.iloc[0]

        else:
            group_0 = n_1_info.gi.iloc[0]
            partition_0 = n_1_info.gj.iloc[0]
            interface_position_0 = n_1_info.y2.iloc[0] - intf_coords[0]
            # interface_position_0 = n_1_info.y2.iloc[0] - intf_coords[0]
            device_id_0 = n_1_info.device_group.iloc[0]

            group_1 = n_0_info.gi.iloc[0]
            partition_1 = n_0_info.gj.iloc[0]
            interface_position_1 = n_0_info.y2.iloc[0] - intf_coords[0]
            # interface_position_1 = n_0_info.y2.iloc[0] - intf_coords[0]
            device_id_1 = n_0_info.device_group.iloc[0]

        # ['group_0', 'partition_0', 'interface_position_0', 'group_1', 'partition_1', 'interface_position_1']
        # unit_type = 'x-unit'
        unit_type = 0   # 'x-unit'

        # for offset in range(0, int(intf_num)):
        #     interface_position_0 += 1
        #     interface_position_1 += 1
        #     intf_data.append((group_0, partition_0, interface_position_0, group_1, partition_1, interface_position_1,
        #                       unit_type, device_group, device_id_0, device_id_1))

        intf_data.append((group_0, partition_0, interface_position_0, group_1, partition_1, interface_position_1,
                          unit_type, device_group, device_id_0, device_id_1, intf_num))

intf_df = pd.DataFrame(intf_data)
intf_df.columns = ['group_0', 'partition_0', 'interface_position_0', 'group_1', 'partition_1', 'interface_position_1',
                   'unit_type', 'device_group', 'device_group_0', 'device_group_1', 'interface_qty']

# Interfaces Summary
intf_desc = intf_df.describe()
intf_df.groupby('device_group').describe()

intf_df.groupby('unit_type').describe()

# data
print(partitions_df.area.sum())
print(intf_df.interface_qty.sum())

print(intf_df.groupby('unit_type').get_group(0).interface_qty.sum())
print(intf_df.groupby('unit_type').get_group(1).interface_qty.sum())

# Export results ---  ---  ---  ---  ---  ---  ---  ---  ---  ---

out_path_base = "/Users/Juan/django_projects/gard/partitions/kl_bisection/post_proc/results"

tmp_path = out_path_base + "/" + model_name
# Remove/Create previous data
delete_folder(tmp_path)
create_folder(tmp_path)

# partitions df
partitions_df.to_csv(tmp_path + "/" + "partitions.csv", index=False)

# Summary group
sg_by_device = summary_groups.groupby("device_group")
for idx, sg_of_device in sg_by_device:
    print("Saving data of device %d" % idx)
    device_tmp_path = tmp_path + "/" + "device_" + str(idx)
    create_folder(device_tmp_path)
    sg_of_device.sort_values(by='n_group').to_csv(device_tmp_path + "/summary_groups.csv", index=False)


# Interfaces
intf_by_device = intf_df.groupby('device_group')
for idx, intf_of_device in intf_by_device:
    print("Saving data interface of device %d" % idx)

    if idx == -1:
        print("global interfaces")
        # intf_of_device.to_csv(tmp_path + "/global_units.csv", index=False)
        intf_by_unit = intf_of_device.groupby('unit_type')
        for idx1, intf_of_unit in intf_by_unit:
            print("Saving interfaces units(0:x-units, 1:y-units): %s" % idx1)
            if idx1 == 0:   # x=units
                intf_of_unit.sort_values(by='group_0').to_csv(tmp_path + "/global_x_units.csv", index=False)
            else:   # y=units
                intf_of_unit.sort_values(by='group_0').to_csv(tmp_path + "/global_y_units.csv", index=False)

    else:
        device_tmp_path = tmp_path + "/" + "device_" + str(idx)
        intf_by_unit = intf_of_device.groupby('unit_type')
        for idx1, intf_of_unit in intf_by_unit:
            print("Saving interfaces units(0:x-units, 1:y-units): %s" % idx1)
            if idx1 == 0:   # x=units
                intf_of_unit.sort_values(by='group_0').to_csv(device_tmp_path + "/x_units.csv", index=False)
            else:   # y=units
                intf_of_unit.sort_values(by='group_0').to_csv(device_tmp_path + "/y_units.csv", index=False)



