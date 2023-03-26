import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import metis

from adabox.legacy.graph.lib.PartitionRectangle import PartitionRectangle
from adabox.legacy.graph.lib.plot_tools import plot_rectangles

# colors_list = list(colors._colors_full_map.values())
colors_list = list(colors.CSS4_COLORS .values())

plt.ion()

base_folder = "/Users/kolibri/PycharmProjects/adaptive-boxes/adabox/decomposition/samples/postdata"

summary_groups_data_path = base_folder + '/' + 'summary_groups.csv'
x_units_path = base_folder + '/' + 'x_units.csv'
y_units_path = base_folder + '/' + 'y_units.csv'
group_details_path = base_folder + '/' + 'group_details.csv'

# Reading Data
group_details = pd.read_csv(group_details_path)
summary_groups = pd.read_csv(summary_groups_data_path)
x_units = pd.read_csv(x_units_path)
y_units = pd.read_csv(y_units_path)

# Getting Codes x_units
codes = []
for iuy in x_units.iterrows():
    g_tmp = iuy[1]['group_0']
    p_tmp = iuy[1]['partition_0']
    code_1 = str(g_tmp) + "_" + str(p_tmp)
    # print(code_1)

    g_tmp = iuy[1]['group_1']
    p_tmp = iuy[1]['partition_1']
    code_2 = str(g_tmp) + "_" + str(p_tmp)
    # print(code_2)
    codes.append((code_1, code_2))

codes_df = pd.DataFrame(codes)

x_units['p0_code'] = codes_df[0]
x_units['p1_code'] = codes_df[1]

# Getting Codes y_units
codes = []
for iuy in y_units.iterrows():
    g_tmp = iuy[1]['group_0']
    p_tmp = iuy[1]['partition_0']
    code_1 = str(g_tmp) + "_" + str(p_tmp)
    # print(code_1)

    g_tmp = iuy[1]['group_1']
    p_tmp = iuy[1]['partition_1']
    code_2 = str(g_tmp) + "_" + str(p_tmp)
    # print(code_2)
    codes.append((code_1, code_2))

codes_df = pd.DataFrame(codes)

y_units['p0_code'] = codes_df[0]
y_units['p1_code'] = codes_df[1]

# getting edges Loop x_units
edges = []
#   x units
color = 0
gs_tmp = x_units.groupby('p0_code')
keys_tmp_level_0 = list(gs_tmp.groups.keys())
for k0 in keys_tmp_level_0:
    print(k0)
    g_tmp = gs_tmp.get_group(k0)

    ggs_tmp = g_tmp.groupby('p1_code')
    keys_tmp_level_1 = list(ggs_tmp.groups.keys())
    for k1 in keys_tmp_level_1:
        print("     %s" % k1)
        weight_tmp = ggs_tmp.groups.get(k1).size
        print("         %d" % weight_tmp)
        edges.append((k0, k1, weight_tmp, color))

#   y units
color = 1
gs_tmp = y_units.groupby('p0_code')
keys_tmp_level_0 = list(gs_tmp.groups.keys())
for k0 in keys_tmp_level_0:
    print(k0)
    g_tmp = gs_tmp.get_group(k0)

    ggs_tmp = g_tmp.groupby('p1_code')
    keys_tmp_level_1 = list(ggs_tmp.groups.keys())
    for k1 in keys_tmp_level_1:
        print("     %s" % k1)
        weight_tmp = ggs_tmp.groups.get(k1).size
        print("         %d" % weight_tmp)
        edges.append((k0, k1, float(weight_tmp), color))

# n_total_nodes = global_keys_no_duplicates.__len__()


#   Creating data frame of edges
edges_df = pd.DataFrame(edges)
edges_df.columns = ['p0', 'p1', 'Weight', 'color']

# Creating Graph
g = nx.Graph()

# Add edges attributes
for i, tmp_row in edges_df.iterrows():
    g.add_edge(tmp_row[0], tmp_row[1], weight=int(tmp_row[2]), attr_dict=tmp_row[2:].to_dict())

n_total_nodes = g.number_of_nodes()
global_keys_no_duplicates = list(g.nodes.keys())

# Getting Node Attributes
nodes_list = []
for k in global_keys_no_duplicates:
    k_indexes_tmp = k.split("_")
    # print(k_indexes_tmp)
    condition_tmp = summary_groups['n_group'] == int(k_indexes_tmp[0])
    smg_tmp = summary_groups[condition_tmp]
    # print(smg_tmp['num_div_x'].iloc[0] * smg_tmp['num_div_y'].iloc[0])

    condition_tmp_loc = (group_details['gi'] == int(k_indexes_tmp[0])) & (group_details['gj'] == int(k_indexes_tmp[1]))
    group_details_tmp = group_details[condition_tmp_loc]
    x_tmp = group_details_tmp["x1"] + abs(group_details_tmp["x2"] - group_details_tmp["x1"]) / 2.0
    y_tmp = group_details_tmp["y1"] + abs(group_details_tmp["y2"] - group_details_tmp["y1"]) / 2.0

    area_tmp = smg_tmp['num_div_x'].iloc[0] * smg_tmp['num_div_y'].iloc[0]
    nodes_list.append(
        (k, int(area_tmp),
         # x_tmp.iloc[0],
         # y_tmp.iloc[0],
         group_details_tmp["x1"].iloc[0],
         group_details_tmp["x2"].iloc[0],
         group_details_tmp["y1"].iloc[0],
         group_details_tmp["y2"].iloc[0],
         x_tmp.iloc[0],
         y_tmp.iloc[0],
         )
    )

nodes_df = pd.DataFrame(nodes_list)
nodes_df.columns = ['code', 'area', 'x1', 'x2', 'y1', 'y2', 'x', 'y']
nodes_df = nodes_df.set_index('code')

# Setting Areas
# nx.set_node_attributes(g, nodes_df.transpose().to_dict())
nx.set_node_attributes(g, nodes_df.to_dict('index'))

print(g.number_of_nodes())
print(g.number_of_edges())

# Define data structure (list) of edge colors for plotting
edge_colors = [e[2]['attr_dict']['color'] for e in g.edges(data=True)]

# Define areas
node_areas = nodes_df.to_dict()['area']

# Define Positions
node_positions = {node[0]: (
node[1]['x1'] + abs(node[1]['x1'] - node[1]['x2']) / 2.0, node[1]['y1'] + abs(node[1]['y1'] - node[1]['y2']) / 2.0) for
                  node in g.nodes(data=True)}

# Plotting
plt.figure(figsize=(8, 6))
nx.draw(g, edge_color=edge_colors, pos=node_positions, node_size=5.0, node_color='black')

# KL-Algorithm
# Graph Partitioning: Recursive Bisection
G = g
# info
print(G.nodes)
print(G.edges)
print(G.nodes(data=True))


def plot_partitions(partitions_arg):
    # Drawing
    partition_list = []
    partitions_tmp = partitions_arg
    com_idx = 0
    for com in partitions_tmp:
        print(com)
        print(com_idx)
        partition_list.extend(list(map(lambda x: (x, {'color': colors_list[com_idx]}), com)))
        com_idx += 1

    # adding colors to dict
    color_nodes_dict = dict(partition_list)
    nx.set_node_attributes(G, color_nodes_dict)
    node_color_map = [n[1]['color'] for n in G.nodes(data=True)]

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


G.graph['edge_weight_attr'] = 'weight'
G.graph['node_weight_attr'] = 'area'

n_parts = 8
(edgecuts, parts) = metis.part_graph(G, n_parts)

# get partitions
nodes_list = np.array(list(G.nodes().keys()))
parts_array = np.array(parts)

partitions = []
for i in range(n_parts):
    print(i)
    partitions.append(list(nodes_list[parts_array == i]))


plot_partitions(partitions)


nx.write_gexf(G, "/Users/Juan/django_projects/gard/partitions/kl_bisection/output/hall/hall_2_partitions.gexf")
