### Regular Import
import numpy as np
import pandas as pd
import random 
import time
import csv
from itertools import permutations
from collections import defaultdict

### Import to run the graph lib
import networkx as nx
from networkx.utils import cuthill_mckee_ordering
from networkx.utils import reverse_cuthill_mckee_ordering

### Import from own-code
from modules.utils import read_Instances
from modules.utils.handle_labels import *
from modules.utils.read_Instances import * 
from modules.utils.read_filenames import *
from modules.graph.Grafo import Grafo, GrafoListaAdj
from modules.centralities.heuristics import *
from modules.components import constructive

def get_centrality_node(graph: nx.Graph, 
                        centrality: dict,) -> dict:
                            

    resp_centrality = centrality["func"](graph, **centrality["args"])
    
    nodes_centrality = {}
    for node, cent_value in resp_centrality.items():
        nodes_centrality[node] = cent_value

    return nodes_centrality

def centrality_heuristic(graph:GrafoListaAdj, centrality_values:dict, cent_str:str,  alpha:float,  iter_max:int, centralities:dict )->int:

    bandwidth = float("inf")
    solution = None

    for _ in range(iter_max):
        solution = constructive.init_Solution_Centrality_lcr(graph=graph, nodes_centrality=centrality_values, random_centrality=cent_str, alpha=alpha, centralities=centralities)
        
        band_solution= Bf_graph(graph=graph, F_labels=solution) 

        if bandwidth > band_solution:
            solution = solution 
            bandwidth = band_solution
            graph_rebuilt = reconstruct_graph_by_labels(graph, solution)

    return bandwidth, graph_rebuilt

