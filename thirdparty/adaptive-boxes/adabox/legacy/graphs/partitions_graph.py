
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph, write_gexf
from matplotlib import pylab


summary_groups_data_path = '/Users/Juan/django_projects/adaptive-boxes/graphs/partitions_data/hall/summary_groups.csv'
x_units_path = '/Users/Juan/django_projects/adaptive-boxes/graphs/partitions_data/hall/x_units.csv'
y_units_path = '/Users/Juan/django_projects/adaptive-boxes/graphs/partitions_data/hall/y_units.csv'


summary_groups = pd.read_csv(summary_groups_data_path)
x_units = pd.read_csv(x_units_path)
y_units = pd.read_csv(y_units_path)


# Creating Graphs
G = nx.Graph()
# n_total_nodes = summary_groups['n_partitions'].sum()
n_total_nodes = summary_groups.shape[0]

H = nx.path_graph(n_total_nodes)
G.add_nodes_from(H)


for idx, row in x_units.iterrows():
    # print(row)
    gi_0 = row['group_0']
    gj_0 = row['partition_0']
    gi_1 = row['group_1']
    gj_1 = row['partition_1']
    G.add_edge(gi_0, gi_1)


for idx, row in y_units.iterrows():
    # print(row)
    gi_0 = row['group_0']
    gj_0 = row['partition_0']
    gi_1 = row['group_1']
    gj_1 = row['partition_1']
    G.add_edge(gi_0, gi_1)


print(G.number_of_nodes())
print(G.number_of_edges())


options = {
     'node_color': 'yellow',
     'node_size': 80,
     'edge_color': 'red',
     'width': 0.5,
     'font_size': 8,
     'font_color': 'black',
}

# save_graph(G, "./my_graph.pdf")

# nx.draw(G, **options)
nx.draw(G, with_labels=True, **options)
plt.show()

nx.write_gexf(G, "/Users/Juan/django_projects/adaptive-boxes/graphs/gexf/hall.gexf")

# Info
print(nx.info(G))
