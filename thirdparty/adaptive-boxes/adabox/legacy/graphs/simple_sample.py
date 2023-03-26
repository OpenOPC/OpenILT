
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

A = np.array([[1, 1], [2, 1]])
G = nx.from_numpy_matrix(A)



print(G.number_of_nodes())
print(G.number_of_edges())

nx.draw(G)
plt.show()


