import torch
from torch_geometric.nn import knn_graph

def build_edges(positions, k=16):
    return knn_graph(positions, k=k, loop=False)