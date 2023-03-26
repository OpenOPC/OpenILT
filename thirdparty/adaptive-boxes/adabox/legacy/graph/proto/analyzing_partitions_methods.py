import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import kernighan_lin_bisection, greedy_modularity_communities, asyn_fluidc
import numpy as np

from adabox.legacy.graph.lib.PartitionRectangle import PartitionRectangle
from adabox.legacy.graph.lib.plot_tools import plot_rectangles

colors_list = list(colors._colors_full_map.values())

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
    g.add_edge(tmp_row[0], tmp_row[1], weight=tmp_row[2], attr_dict=tmp_row[2:].to_dict())

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
        (k, area_tmp,
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
nx.set_node_attributes(g, nodes_df.transpose().to_dict())

print(g.number_of_nodes())
print(g.number_of_edges())


# Define data structure (list) of edge colors for plotting
edge_colors = [e[2]['attr_dict']['color'] for e in g.edges(data=True)]


# Define areas
node_areas = nodes_df.to_dict()['area']

# Define Positions
node_positions = {node[0]: (node[1]['x1'] + abs(node[1]['x1'] - node[1]['x2'])/2.0, node[1]['y1'] + abs(node[1]['y1'] - node[1]['y2'])/2.0) for node in g.nodes(data=True)}


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


# Recursive bisection -  First Run (resulting in 2 partitions)
partitions = kernighan_lin_bisection(G, weight='Weight')
plot_partitions(partitions)

SG1 = G.subgraph(list(partitions[0]))
sub_partitions_1 = kernighan_lin_bisection(SG1, weight='Weight')
plot_partitions(sub_partitions_1)

# Loop
for i in range(0, 1):
    pss = []
    for p in partitions:
        ps_tmp = kernighan_lin_bisection(G.subgraph(list(p)), weight='Weight')
        pss.extend(ps_tmp)

    partitions = pss

plot_partitions(partitions)


partitions = kernighan_lin_bisection(G)
# Loop
for i in range(0, 1):
    pss = []
    for p in partitions:
        ps_tmp = kernighan_lin_bisection(G.subgraph(list(p)))
        pss.extend(ps_tmp)

    partitions = pss

plot_partitions(partitions)


# Modularity-based communities I: greedy_modularity_communities
partitions = greedy_modularity_communities(G, weight='Weight', best_n=4, resolution=1000000)
plot_partitions(partitions)


partitions = greedy_modularity_communities(G, best_n=8, resolution=1000000000000000000000)
plot_partitions(partitions)

# Fluid Communities
partitions = asyn_fluidc(G, k=8, max_iter=100)
plot_partitions(partitions)


# Get stats from partitions --------
SCALE = 1
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


