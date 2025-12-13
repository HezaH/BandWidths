import numpy as np
import random
from ..utils import handle_labels 
from ..graph.Grafo import Grafo, GrafoListaAdj
import time
import networkx as nx


def initialSolution(graph: GrafoListaAdj)->dict:
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


def initialSolutionCentralidade(graph: GrafoListaAdj, edges:list, heuristic)->dict:
    
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

def initialSolutionMultiNivel(graph: GrafoListaAdj, nodes_centrality_eigenvector:dict)->dict:
    
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

   
    random_centrality = nodes_centrality_eigenvector 

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