# lendo todas as instancias de uma classe

import os
import time
import pandas as pd
import networkx as nx
import torch
import numpy as np
from modules.utils.read_filenames import readFilesInDict
from modules.graph.Grafo import GrafoListaAdj
from centralities import get_centrality_node, centrality_heuristic
from agent import Agent
from enviroment import Env
from modules.utils import read_Instances
from modules.centralities.heuristics import biggest_eigenvector, reverse_cuthill
from modules.components import constructive
import matplotlib.pyplot as plt
from modules.VNS import local_search, shakes, init_solution
from modules.utils.handle_labels import Bf_graph

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

def listar_instancias(path):
    instancias = []
    for pasta_atual, subpastas, arquivos in os.walk(path):
        for arquivo in arquivos:
            # Aqui você pode adicionar lógica para filtrar os arquivos, se necessário
            caminho_completo = os.path.join(pasta_atual, arquivo)
            instancias.append(caminho_completo)
    return instancias

dir_list = ["computational_fluid_dynamics", "electromagnetics", "optimization", "structural"]
for loop in range(1,2):

    filename = './result_output_cuthill_autovetor_maior.csv'
    if loop == 1:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(filename)

    list_instance = []
    list_band = []
    list_time = []

    for kind in dir_list:
        main_path = f'./newdata/{kind}'
        list_path = readFilesInDict(main_path, ".mtx")

        for instancia in list_path:

            print( "####### instancia", instancia )
            nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance(instancia)
            
            #parametros
            max_iter = 100
            # centralities = {0:"eigenvector", 1:"degree", 2:"closeness"}
            centralities = {0:"eigenvector", 1:"degree"}
            todos_movimentos = list(range(len(centralities)))

            #instancia os graficos networkx
            G = nx.Graph()
            G.add_edges_from(edges)

            #instancia os graficos own-lib
            grafo_adj = GrafoListaAdj()
            grafo_adj.DefinirN(nnodes,VizinhancaDuplamenteLigada=True)
            for (u, v) in edges:
                grafo_adj.AdicionarAresta(u, v)

            #dict of centralities values for each centrality
            centralities_maps={}
            for centrality in centralities.values():
                centralities_maps[centrality] = get_centrality_node(G,  centrality)
            
            #Agent to learning and trainning
            agent = Agent(learning_rate=0.001,
                        gamma=0.9,
                        epsilon=1,
                        eps_min=0.01,
                        eps_dec=0.995,
                        n_movements=len(centralities),
                        n_actions=len(todos_movimentos),
                        n_states=4,
                        deep=True)

            env = Env(7990)

            state = env.get_initial_state()
            agent_movements = agent.get_movements()
            score = 0
            t=0

            for i in range(max_iter):
                env.reset()
                t = i+2
                #Escolha da centralidade e da lista
                action = agent.choose_action(state)
                centrality = centralities[action]

                centrality_dict_values = centralities_maps[centrality]
                
                info = env.step(graph=grafo_adj, centrality_values=centrality_dict_values, cent_str=centrality)
                new_state = [info["n_steps"], info["gap"], info["reward"], info["bandwidth"]]

                reward = info["reward"]
                gap = info["gap"]
                bandwidth = info["bandwidth"]

                #Treinamento do novo estado
                agent.learn(state, action, reward, new_state)

                #score acumulado por iteração
                score += reward

                print(f">> Episode {i+1} Gap: {gap} Reward: {reward} Score: {score} Centrality: {centrality} Bandwidth: {bandwidth}")
                state = new_state
            
            torch.save(agent.Q.state_dict(), 'trained_model_{}.pth')


            band = bandwidth(G)
            band_after = reverse_cuthill(G, biggest_eigenvector)
            print("##### Biggest Eigenvector ########")
            print("Banda original: ", band)
            print("Banda Final: ", band_after)

            centralities = {0:"eigenvector", 1:"degree"}
            todos_movimentos = list(range(len(centralities)))

            custo_s = centrality_heuristic(graph=grafo_adj, centrality_values=centralities_maps[centralities[0]], cent_str=centralities[0], alpha=0.3, iter_max=50)
            print("Banda Final multicetrality: ", custo_s)

nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance("./newdata/structural/bcsstk05/bcsstk05.mtx")

# Exemplo de criação de um grafo
G = nx.Graph()
G.add_edges_from(edges)

def bandwidth(graph):
    bw = 0
    for u, v in graph.edges():
        bw = max(bw, abs(u - v))
    return bw 

original_bandwidth = bandwidth(G)


import networkx as nx
nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance("./newdata/structural/bcsstk02/bcsstk02.mtx")

# Exemplo de criação de um grafo
G = nx.Graph()
G.add_edges_from(edges)

# Aplicar o algoritmo Cuthill-McKee reverso
rcm_order = list(nx.utils.reverse_cuthill_mckee_ordering(G))

# Criar um novo grafo com a ordem dos nós reordenada
G_reordered = nx.relabel_nodes(G, {old_label: new_label for new_label, old_label in enumerate(rcm_order)})

# Exibir o grafo original e o grafo reordenado
print("Grafo original:")
print(G.edges())

print("\nOrdem dos nós pelo algoritmo Cuthill-McKee reverso:")
print(rcm_order)

print("\nGrafo reordenado:")
print(G_reordered.edges())

# Função para calcular a largura de banda do grafo
def bandwidth(graph):
    bw = 0
    for u, v in graph.edges():
        bw = max(bw, abs(u - v))
    return bw

# Calcular a largura de banda dos grafos
original_bandwidth = bandwidth(G)
reordered_bandwidth = bandwidth(G_reordered)

print(f"\nLargura de banda original: {original_bandwidth}")
print(f"Largura de banda reordenada: {reordered_bandwidth}")


#instancia os graficos 
grafo_adj = GrafoListaAdj()
grafo_adj.DefinirN(nnodes,VizinhancaDuplamenteLigada=True)
for (u, v) in edges:
    grafo_adj.AdicionarAresta(u, v)

#dict of centralities values for each centrality
centralities_maps={}
for centrality in centralities.values():
    centralities_maps[centrality] = get_centrality_node(G,  centrality)

solution = constructive.init_Solution_Centrality_lcr(graph=grafo_adj, nodes_centrality=centralities_maps["degree"], random_centrality="degree", alpha=0.45)


G_reordered = nx.relabel_nodes(G, solution)

adj_matrix_reordered = nx.to_scipy_sparse_matrix(G_reordered, format='csr')

# Função para plotar uma matriz esparsa
def plot_sparse_matrix(matrix, title):
    plt.figure(figsize=(8, 8))
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.show()

plot_sparse_matrix(adj_matrix_reordered, "matrix")

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