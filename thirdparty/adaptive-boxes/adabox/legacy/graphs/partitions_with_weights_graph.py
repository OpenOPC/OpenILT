
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from networkx.readwrite import json_graph, write_gexf
from matplotlib import pylab


summary_groups_data_path = '/Users/Juan/django_projects/adaptive-boxes/graphs/partitions_data/boston/summary_groups.csv'
x_units_path = '/Users/Juan/django_projects/adaptive-boxes/graphs/partitions_data/boston/x_units.csv'
y_units_path = '/Users/Juan/django_projects/adaptive-boxes/graphs/partitions_data/boston/y_units.csv'


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


# Getting al keys
gs_tmp = x_units.groupby('p0_code')
p0_keys_tmp = list(gs_tmp.groups.keys())

gs_tmp = x_units.groupby('p1_code')
p1_keys_tmp = list(gs_tmp.groups.keys())

global_keys = []
global_keys.extend(p0_keys_tmp)
global_keys.extend(p1_keys_tmp)


def remove_duplicates(l):
    return list(set(l))


global_keys_no_duplicates = remove_duplicates(global_keys)


# Getting edges x_units
gs_tmp = x_units.groupby('p0_code')
keys_tmp_level_0 = list(gs_tmp.groups.keys())
g_tmp = gs_tmp.get_group(keys_tmp_level_0[0])


ggs_tmp = g_tmp.groupby('p1_code')
keys_tmp_level_1 = list(ggs_tmp.groups.keys())
weight_tmp = ggs_tmp.groups.get(keys_tmp_level_1[0]).size
print(weight_tmp)


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


n_total_nodes = global_keys_no_duplicates.__len__()


#   Creating data frame of edges
edges_df = pd.DataFrame(edges)
edges_df.columns = ['p0', 'p1', 'Weight', 'color']


# Getting Node Attributes
nodes_list = []
for k in global_keys_no_duplicates:
    k_indexes_tmp = k.split("_")
    # print(k_indexes_tmp)
    condition_tmp = summary_groups['n_group'] == int(k_indexes_tmp[0])
    smg_tmp = summary_groups[condition_tmp]
    # print(smg_tmp['num_div_x'].iloc[0] * smg_tmp['num_div_y'].iloc[0])
    area_tmp = smg_tmp['num_div_x'].iloc[0] * smg_tmp['num_div_y'].iloc[0]
    nodes_list.append((k, area_tmp))


nodes_df = pd.DataFrame(nodes_list)
nodes_df.columns = ['code', 'area']
nodes_df = nodes_df.set_index('code')


# Creating Graph
g = nx.Graph()

# Add edges attributes
for i, tmp_row in edges_df.iterrows():
    g.add_edge(tmp_row[0], tmp_row[1], weight=tmp_row[2] ,attr_dict=tmp_row[2:].to_dict())


# Setting Areas
nx.set_node_attributes(g, nodes_df.transpose().to_dict())


print(g.number_of_nodes())
print(g.number_of_edges())


# Define data structure (list) of edge colors for plotting
edge_colors = [e[2]['attr_dict']['color'] for e in g.edges(data=True)]


# Define areas
node_areas = nodes_df.to_dict()['area']

# Plotting
plt.figure(figsize=(8, 6))
nx.draw(g, edge_color=edge_colors, node_size=10, node_color='black')

# nx.draw_shell(g, edge_color=edge_colors, node_size=10, node_color='black')
plt.title('Graph Representation of Sleeping Giant Trail Map', size=15)
plt.show()

# Save
nx.write_gexf(g, "/Users/Juan/django_projects/adaptive-boxes/graphs/gexf/boston.gexf")
