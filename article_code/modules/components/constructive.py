#regular imports
import numpy as np
import random
import time
import networkx as nx

#import models
from modules.utils import handle_labels
from modules.graph.Grafo import Grafo, GrafoListaAdj

def get_LCR(queue:list, chosen_centrality:list, random_centrality:str, alpha:float):
    
    reverse_centralities = {"eigenvector": True, "degree": False, "closeness": False }

    vertices_ordenados = sorted(queue, key=lambda x: chosen_centrality[x], reverse=reverse_centralities[random_centrality])
    qtd_elem = int(len(vertices_ordenados) * alpha)
    LCR = vertices_ordenados[:qtd_elem]
    LCR_comp = vertices_ordenados[qtd_elem:]
    random.shuffle(LCR)
    random.shuffle(LCR_comp)
    vertices_orden_rotulagem = LCR + LCR_comp

    return vertices_orden_rotulagem

def init_Solution(graph: GrafoListaAdj)->dict:
    ''''FUNCIONANDO PROFUNDAMENTE'''
    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 #O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #O primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0 #
    
    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        h = random.randint(0, ql-1)
        
        for j1 in range(ql):
            n = queue[h]
            l = l + 1
            f[n] = l
            h = h + 1
            if h >= ql:
                h = 0
        
        queue = stack
        ql = r
        
    return f


def init_Solution_Centrality(graph: GrafoListaAdj, edges:list, heuristic)->dict:
    
    # calculando a centralidade de cada vertice
    if heuristic != None:
        G = nx.Graph()
        G.add_edges_from(edges)
        resp_centrality = heuristic[0](G)
        nodes_centrality = {}
        for node, centrality in resp_centrality.items():
            nodes_centrality[node] = centrality
    

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        
        
        vertices_ordenados = sorted(queue, key=lambda x: nodes_centrality[x], reverse=heuristic[1])
        
        h = 0
        for j1 in range(ql):
            n = vertices_ordenados[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f

def init_Solution_Centrality_otm(graph: GrafoListaAdj, nodes_centrality:dict)->dict:
    
    # calculando a centralidade de cada vertice

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    random_centrality = nodes_centrality 

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        vertices_ordenados = sorted(queue, key=lambda x: random_centrality[x], reverse=True)
        
        
        h = 0
        for j1 in range(ql):
            n = vertices_ordenados[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f

def init_Solution_Centrality_lcr(graph: GrafoListaAdj,  nodes_centrality:dict, random_centrality:str, alpha=float)->dict:
    
    # calculando a centralidade de cada vertice

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)
    
    # Calcular as probabilidades de cada vértice
    sum_values = sum(nodes_centrality.values())
    probs = [valor/sum_values for valor in nodes_centrality.values()]

    # Sortear um item baseado nas probabilidades
    k = random.choices(list(nodes_centrality.keys()), probs)[0] 
    
    # k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice assinado nno v_zero
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        array_LCR = get_LCR(queue=queue, chosen_centrality=nodes_centrality, random_centrality=random_centrality, alpha=alpha)
        
        h = 0
        for j1 in range(ql):
            n = array_LCR[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f