#### Import to read the files dict 
import os
from os import listdir
from os.path import isfile, join
from os import error

import datetime
import numpy as np
from numpy import arange

import math
import csv

from collections import defaultdict

###### measure bandwidth  ########

def labels(R):
    f = defaultdict(int)
    label = 1
    for node in R:
        f[node] = label
        label=label+1
    return f

def measureBand(adj_dict, F):
    width = 0
    for key in adj_dict:
        for node in adj_dict[key]:
            temp = abs(F[key] - F[node])
            if temp > width:
                width = temp
            
    return width

### Import to run the graph lib
import networkx as nx
from networkx.utils import cuthill_mckee_ordering
from networkx.utils import reverse_cuthill_mckee_ordering

#HEURISTIC SETTING

# HEURISTIC BIGGEST DEGREE
def biggest_degree(G):
    return max(G, key=G.degree)

# HEURISTIC SMALLEST DEGREE
def smallest_degree(G):
    return min(G, key=G.degree)

# # HEURISTIC BIGGEST EIGENVECTOR
# def smallest_eigenvector(G):
#     centrality = nx.eigenvector_centrality(G, max_iter=600)
#     return max(centrality, key=centrality.get)

# # HEURISTIC SMALLEST EIGENVECTOR
# def biggest_eigenvector(G):
#     centrality = nx.eigenvector_centrality(G, max_iter=600)
#     return min(centrality, key=centrality.get)

# # HEURISTIC BIGGEST KATZ
# def biggest_katz(G):
#     centrality = nx.katz_centrality(G, max_iter=100000)
#     return max(centrality, key=centrality.get)

# # HEURISTIC SMALLEST KATZ
# def smallest_katz(G):
#     centrality = nx.katz_centrality(G, max_iter=100000)
#     return min(centrality, key=centrality.get)

# HEURISTIC BIGGEST EIGENVECTOR NUMPY
def smallest_eigenvector(G):
    centrality = nx.eigenvector_centrality_numpy(G)
    return max(centrality, key=centrality.get)

# HEURISTIC SMALLEST EIGENVECTOR NUMPY
def biggest_eigenvector(G):
    centrality = nx.eigenvector_centrality_numpy(G)
    return min(centrality, key=centrality.get)

# HEURISTIC BIGGEST KATZ NUMPY
def biggest_katz(G):
    centrality =  (G)
    return max(centrality, key=centrality.get)

# HEURISTIC SMALLEST KATZ NUMPY
def smallest_katz(G):
    centrality = nx.katz_centrality_numpy(G)
    return min(centrality, key=centrality.get)

# HEURISTIC BIGGEST CLOSENESS (Shortest Path)
def biggest_closeness(G):
    centrality = nx.closeness_centrality(G)
    return max(centrality, key=centrality.get)

# HEURISTIC SMALLEST CLOSENESS (Shortest Path)
def smallest_closeness(G):
    centrality = nx.closeness_centrality(G)
    return min(centrality, key=centrality.get)

# HEURISTIC BIGGEST HARMONIC
def biggest_harmonic(G):
    centrality = nx.harmonic_centrality(G)
    return max(centrality, key=centrality.get)

# HEURISTIC SMALLEST HARMONIC
def smallest_harmonic(G):
    centrality = nx.harmonic_centrality(G)
    return min(centrality, key=centrality.get)

#HEURISTIC BIGGEST BETWEENNESS
def biggest_betweenness(G):
    centrality = nx.betweenness_centrality(G)
    return max(centrality, key=centrality.get)

#HEURISTIC SMALLEST BETWEENNESS
def smallest_betweenness(G):
    centrality = nx.betweenness_centrality(G)
    return min(centrality, key=centrality.get)

#HEURISTIC BIGGEST PERCOLATION
def biggest_percolation(G):
    centrality = nx.percolation_centrality(G)
    return max(centrality, key=centrality.get)
    
#HEURISTIC SMALLEST PERCOLATION
def smallest_percolation(G):
    centrality = nx.percolation_centrality(G)
    return min(centrality, key=centrality.get)

#FUNCTION TO GET BANDWIDTH 
def bandwidth(G):
    '''Calculate the bandwidth'''
    A = nx.adjacency_matrix(G)
    x, y = np.nonzero(A)
    w = (y - x).max() + (x - y).max() + 1
    return w

#INIT A GRAPH
def init_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

# BANDWIDTH CUTHILL MCKEE
def cuthill(G, heuristic):
    rcm = list(cuthill_mckee_ordering(G, heuristic=heuristic))
    adj_matrix = nx.adjacency_matrix(G, nodelist=rcm)
    x, y = np.nonzero(adj_matrix)
    return (y - x).max() + (x - y).max() + 1

# BANDWIDTH REVERSE CUTHILL MCKEE
def reverse_cuthill(G, heuristic):
    rcm = list(reverse_cuthill_mckee_ordering(G, heuristic=heuristic))
    adj_matrix = nx.adjacency_matrix(G, nodelist=rcm)
    x, y = np.nonzero(adj_matrix)
    return (y - x).max() + (x - y).max() + 1