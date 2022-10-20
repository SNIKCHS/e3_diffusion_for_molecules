import argparse
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import community as community_louvain
# for ARI computation
from sklearn import preprocessing, metrics
import GraphRicciCurvature

from qm9 import dataset

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff

def get_adj_matrix(n_nodes):
    # 对每个n_nodes，batch_size只要算一次

    # get edges for a single sample
    rows, cols = [], []

    for i in range(n_nodes):
        for j in range(n_nodes):
            rows.append(i)
            cols.append(j)
    edges = [torch.LongTensor(rows),
             torch.LongTensor(cols)]
    return edges

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--include_charges', type=eval, default=False,
                    help='include atom charge or not')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--remove_h', action='store_true')

args = parser.parse_args()
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
atom = next(iter(dataloaders['train']))

x = atom['positions'] # b,n_nodes,3
b,n_nodes,dim = x.size()
x = x.view(-1,dim)
node_mask = atom['atom_mask'].unsqueeze(2)
edge_mask = atom['edge_mask']
one_hot = atom['one_hot']
categories = (torch.argmax(one_hot.int(), dim=2) + 1) * node_mask.squeeze() # b,n_nodes

atom_charge_dict = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
edge = get_adj_matrix(n_nodes)
dist,_ = coord2diff(x,edge)

G = nx.Graph()
for i in range(len(edge[0])):
    a = edge[0][i]
    b = edge[1][i]
    if a<b and categories[0,a]!=0 and categories[0,b]!=0:
        a_type = atom_charge_dict[categories[0,a].item()]+str(a.item())
        b_type = atom_charge_dict[categories[0,b].item()]+str(b.item())
        print(a_type, b_type, dist[i].item())
        G.add_edge(a_type, b_type, dist=dist[i].item())


# G = nx.Graph()
#
# G.add_edge('H1', 'O', dist=0.958)
# G.add_edge('H2', 'O', dist=0.958)
# G.add_edge('H1', 'H2', dist=1.515)

# for (n1, n2, d) in G.edges(data=True):
#     d.clear()   # remove edge weight
print(nx.info(G))

orc = OllivierRicci(G, weight='dist', alpha=0.5, verbose="TRACE")

orc.compute_ricci_curvature()
G_orc = orc.G.copy()

pos_nodes = nx.spring_layout(G_orc)
nx.draw(G_orc, pos_nodes, with_labels=True)
pos_attr = {}
for n, coor in pos_nodes.items():
    pos_attr[n] = (coor[0], coor[1] - 0.08)
# print(nx.get_node_attributes())
node_attr = nx.get_node_attributes(G_orc, "ricciCurvature")
print(node_attr)
cus_node_attr = {}
for n, attr in node_attr.items():
    cus_node_attr[n] = "c : %.2f" % (attr)
nx.draw_networkx_labels(G, pos=pos_attr, labels=cus_node_attr)
plt.savefig('graph.svg')


def show_results(G, curvature="ricciCurvature"):
    # Print the first five results
    print("Karate Club Graph, first 5 edges: ")

    for n1, n2 in list(G.edges()):
        print("Ricci curvature of edge (%s,%s) is %f" % (n1, n2, G[n1][n2][curvature]))
    for n in list(G.nodes()):
        print("Ricci curvature of node (%s) is %f" % (n, G.nodes[n][curvature]))

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures, bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures (Karate Club)")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights, bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights (Karate Club)")

    plt.tight_layout()
    plt.savefig('curvtures.svg', dpi=None)


show_results(G_orc)
