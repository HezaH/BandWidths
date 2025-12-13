from utils import read_Instances
from grafo.Grafo import Grafo, GrafoListaAdj
import random 
from utils.handle_labels import *
from utils.read_Instances import *
from utils.read_filenames import *
from collections import defaultdict
from VNS import local_search, shakes, init_solution
import time
import csv
from centralidades.construtivos import *
import pandas as pd

### Import to run the graph lib
import networkx as nx
from networkx.utils import cuthill_mckee_ordering
from networkx.utils import reverse_cuthill_mckee_ordering

#INIT A GRAPH by networkx
def init_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

#FUNCTION TO GET BANDWIDTH  by networkx
def bandwidth(G):
    '''Calculate the bandwidth'''
    A = nx.adjacency_matrix(G)
    x, y = np.nonzero(A)
    
    w = (y - x).max() + (x - y).max() + 1
    return w


def VnsBand(graph: GrafoListaAdj, k_min:int, k_max:int, k_step:int, t_max:int, alpha:int)->dict:
    Bandwidth = float("inf")
    init_temp = time.time()
    delta_temp = 0 
    i_max = int((k_max-k_min)/k_step)
    solution_f_star = None
    # count = 1
    
    while delta_temp < t_max:
        # print("loop 1: ", count)
        solution_f = init_solution.initialSolution(graph=graph)
        solution_f = local_search.improved_hill_climbing(graph=graph, solution_f=solution_f)
        i = 0
        k = k_min
        count_f = 0
        while i<=i_max:
            
            count_f = count_f + 1
            # print("loop 2: ", count_f)
            # print("i : ", i)
            # print("i_max : ", i_max)
            
            solution_f_l = shakes.control(graph=graph, solution_f=solution_f, k=k, k_min=k_min, k_step=k_step)
            solution_f_l = local_search.improved_hill_climbing(graph=graph, solution_f=solution_f_l)
            
            if move(graph=graph, solution_f1=solution_f, solution_f2=solution_f_l, alpha=alpha):
                solution_f = solution_f_l.copy()
                k = k_min
                i = 0
            else:
                k = k + k_step
                i = i + 1
        Bandwidth_f = Bf_graph(graph=graph, F_labels=solution_f)
        if Bandwidth_f < Bandwidth:
             Bandwidth = Bandwidth_f
             solution_f_star = solution_f.copy()
        
        # count = count + 1 
        
        now_temp= time.time()
        delta_temp = now_temp - init_temp
    
    return solution_f_star

def VnsBand_multistart(graph: GrafoListaAdj, edges:list, k_min:int, k_max:int, k_step:int, t_max:int, alpha:int)->dict:

    heuristics = [ biggest_degree, smallest_degree, biggest_eigenvector, smallest_eigenvector, biggest_katz, smallest_katz,
                    biggest_closeness, smallest_closeness, biggest_harmonic, smallest_harmonic,
                    biggest_betweenness, smallest_betweenness, biggest_percolation, smallest_percolation ]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    labels_before = {node: node for node in G.nodes}

    Bandwidth = float("inf")
    init_temp = time.time()
    delta_temp = 0 
    i_max = int((k_max-k_min)/k_step)
    solution_f_star = None
    count = 0
    
    while delta_temp < t_max:
        count = count + 1

        solution_f = None

        if count%2==0:
            solution_f = init_solution.initialSolution(graph=graph)
            solution_f = local_search.improved_hill_climbing(graph=graph, solution_f=solution_f)
        else:
            random_heuristic = random.choice(heuristics)
            permutation = cuthill_mckee_ordering(G, heuristic=random_heuristic)
            # Reorganize os rótulos de acordo com a permutação
            labels_after = {i+1: labels_before[node] for i, node in enumerate(permutation)}
            solution_f = local_search.improved_hill_climbing(graph=graph, solution_f=labels_after)

        i = 0
        k = k_min
        count_f = 0
        while i<=i_max:
            
            count_f = count_f + 1
            # print("loop 2: ", count_f)
            # print("i : ", i)
            # print("i_max : ", i_max)
            
            solution_f_l = shakes.shake_two(graph=graph, solution_f=solution_f, k=k)
            solution_f_l = local_search.improved_hill_climbing(graph=graph, solution_f=solution_f_l)
            
            if move(graph=graph, solution_f1=solution_f, solution_f2=solution_f_l, alpha=alpha):
                solution_f = solution_f_l.copy()
                k = k_min
                i = 0
            else:
                k = k + k_step
                i = i + 1
        Bandwidth_f = Bf_graph(graph=graph, F_labels=solution_f)
        if Bandwidth_f < Bandwidth:
             Bandwidth = Bandwidth_f
             solution_f_star = solution_f.copy()
        
        # count = count + 1 
        
        now_temp= time.time()
        delta_temp = now_temp - init_temp
    
    return solution_f_star

def smallest_eigenvector(G):
    centrality = nx.eigenvector_centrality_numpy(G)
    return max(centrality, key=centrality.get)

# HEURISTIC SMALLEST EIGENVECTOR NUMPY
def biggest_eigenvector(G):
    centrality = nx.eigenvector_centrality_numpy(G)
    return min(centrality, key=centrality.get)

        

def main():
    # path = "./data/bp_200.mtx"
    # nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance(path)

    # print("Instance: ", path)

    # #Grafo usando a estrutura do networkx
    # G = nx.Graph()
    # G.add_edges_from(edges)

    # original_w = bandwidth(G)

    # print("BF inicial: ", original_w)

    # grafo_adj = GrafoListaAdj()
    # grafo_adj.DefinirN(nnodes,VizinhancaDuplamenteLigada=True)
    
    # for (u, v) in edges:
    #     grafo_adj.AdicionarAresta(u, v)

    # # vns_solution = VnsBand_multistart(graph=grafo_adj, edges=edges, k_min=5, k_max=220, k_step=5, t_max=360, alpha=10)
    # # solution = init_solution.initialSolutionCentralidade(graph=grafo_adj, edges=edges, heuristic=nx.eigenvector_centrality_numpy)
    
    # solution = init_solution.initialSolutionMultiNivel(graph=grafo_adj, edges=edges)
    # band_solution= Bf_graph(grafo_adj, solution)
    
    # print("BF final: ", band_solution)

    main_path = "/Users/jvmaues/Documents/OTIMIZACAO/TCC-OFICIAL/CODIGOS/REFAZENDO/data"
    
    list_path = readFilesInDict(main_path, ".mtx")

    total = len(list_path)
    instance_n = 1


    for loop in range(1,2):

        filename = './result_output_cuthill_autovetor_maior.csv'

        # fieldnames = ['instance', f'band{loop}', f'time{loop}']

        if loop == 1:
            df = pd.DataFrame()
        else:
            df = pd.read_csv(filename)
        
        list_instance = []
        list_band = []
        list_time = []

            
        for instance_path in list_path:

            instance_name = instance_path.replace(main_path, "").replace("/", "").replace(".mtx", "")

            print("#######", instance_name ,"# instance: ", instance_n, " of ", total,"\n", end='\r')
            instance_n = instance_n + 1 
            
            nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance(instance_path)

            #instancia os graficos
            grafo_adj = GrafoListaAdj()
            grafo_adj.DefinirN(nnodes,VizinhancaDuplamenteLigada=True)
            for (u, v) in edges:
                grafo_adj.AdicionarAresta(u, v)

            try:
                band = float("inf")
                solution = None
                
                init = time.time()
                G = nx.Graph()
                G.add_edges_from(edges)
                labels_before = {node: node for node in G.nodes}

                resp_centrality_eigenvector = nx.eigenvector_centrality_numpy(G)
                nodes_centrality_eigenvector = {}
                for node, centrality in resp_centrality_eigenvector.items():
                    nodes_centrality_eigenvector[node] = centrality

                resp_centrality_degree = nx.degree_centrality(G)
                nodes_centrality_degree = {}
                for node, centrality in resp_centrality_degree.items():
                    nodes_centrality_degree[node] = centrality

                rcm = nx.utils.cuthill_mckee_ordering(G, heuristic=biggest_eigenvector)
                labels_after = {i+1: labels_before[node] for i, node in enumerate(rcm)}
                for i in range(300):  
                    # heuristics = [ (nx.degree_centrality, False), (nx.eigenvector_centrality_numpy, False)]
                    # random_centrality = random.choice(heuristics)
                    solution = init_solution.initialSolution(graph=grafo_adj)
                    # solution = init_solution.initialSolutionMultiNivel(graph=grafo_adj,nodes_centrality_eigenvector=nodes_centrality_degree )
                band_solution= Bf_graph(grafo_adj, labels_after)



                if band > band_solution:
                    solution = solution 
                    band = band_solution

                final = time.time()
                delta = final - init

                print("BF final: ", band)

            except:
                delta = "error"
                band = "error"

            finally:
                list_instance.append(instance_name)
                list_time.append(delta)
                list_band.append(band)
                
        if loop == 1:
            df['instance'] = list_instance
            df['band0'] = list_band
            df['time0'] = list_time

        else:
            df[f'band{loop}'] = list_band
            df[f'time{loop}'] = list_time
        
        df.to_csv(filename, index=False)
       
                
if __name__ == "__main__":
	main()