import numpy as np
import random

from ..utils import handle_labels 
from ..graph.Grafo import Grafo, GrafoListaAdj
import time
from collections import defaultdict

   
def hill_climbing(graph: GrafoListaAdj, solution_f: dict)->dict:
    '''
    
    '''
    solution_current_f = solution_f.copy()
    can_improve = True
    Bf_global_current = handle_labels.Bf_graph(graph, solution_current_f)
    count = 0
    while can_improve == True:
        can_improve = False
        for node_v in graph.V():
            Bf_node_current = handle_labels.Bf_node(graph, solution_current_f, node_v)
            # print("passing node_v: ", node_v)
            if Bf_global_current == Bf_node_current:
                mid_neighborhood = handle_labels.get_mid_neighborhood(graph, solution_current_f, node_v)
                for node_u in mid_neighborhood:
                    # print("passing node_u: ", node_u)
                    n_critical_edges_before = handle_labels.get_n_critical_edges(graph, solution_current_f, node_u, node_v)
                    # print("passing critical edges before: ", n_critical_edges_before)
                    solution_current_f = handle_labels.swapping(solution_f, node_v, node_u)
                    Bf_global_current = handle_labels.Bf_graph(graph, solution_current_f) # Seria interessante arranjar uma maneira de atualizar apenas o Bf dos vizinhos de 
                                                                                          # node_u com os vizinhos de node_v de maneira mais eficiente
                    n_critical_edges_now = handle_labels.get_n_critical_edges(graph, solution_current_f, node_u, node_v)
                    if n_critical_edges_now < n_critical_edges_before:
                        # print("passing critical edges now: ", n_critical_edges_now)
                        can_improve = True
                        break
                    solution_current_f = handle_labels.swapping(solution_f, node_v, node_u)
                    Bf_global_current = handle_labels.Bf_graph(graph, solution_current_f)
                    # print("passing on last for...")
        count = count + 1
    
    return solution_current_f


def improved_hill_climbing(graph: GrafoListaAdj, solution_f: dict)->dict:
    '''
    
    '''
    solution_current_f = solution_f
    can_improve = True
    Bf_global_current = handle_labels.Bf_graph(graph, solution_current_f)
    count = 0
    count_v = 0
    
    while can_improve == True:
        can_improve = False
        for node_v in graph.V():
            Bf_node_current = handle_labels.Bf_node(graph, solution_current_f, node_v)
            
            if Bf_global_current == Bf_node_current:
                mid_neighborhood = handle_labels.get_mid_neighborhood(graph, solution_current_f, node_v)
                for node_u in mid_neighborhood:
                    
                    # n_critical_edges_before_node_v = handle_labels.get_n_critical_edges_by_node(graph, solution_current_f, node_v, Bf_global_current)
                    # n_critical_edges_before_node_u = handle_labels.get_n_critical_edges_by_node(graph, solution_current_f, node_u, Bf_global_current)

                    n_critical_edges_before_node_v = handle_labels.Vc(graph, solution_current_f)
                    n_critical_edges_before_node_u = handle_labels.Vc(graph, solution_current_f)

                    solution_test = handle_labels.swapping(solution_f, node_v, node_u)

                    # n_critical_edges_now_node_v = handle_labels.get_n_critical_edges_by_node(graph, solution_test, node_v, Bf_global_current)
                    # n_critical_edges_now_node_u = handle_labels.get_n_critical_edges_by_node(graph, solution_test, node_u, Bf_global_current)

                    n_critical_edges_now_node_v = handle_labels.Vc(graph, solution_test)
                    n_critical_edges_now_node_u = handle_labels.Vc(graph, solution_test)

                    if (n_critical_edges_before_node_u <= n_critical_edges_now_node_u) and ( n_critical_edges_before_node_v <= n_critical_edges_now_node_v)and(n_critical_edges_before_node_u + n_critical_edges_before_node_v < n_critical_edges_now_node_u + n_critical_edges_now_node_v):
                        can_improve = True
                        solution_current_f = solution_test
                        Bf_global_current = handle_labels.Bf_graph(graph, solution_current_f)
                        break

    return solution_current_f

def improved_hill_climbing_two(graph: GrafoListaAdj, solution_f: dict)->dict:
    N = graph.n
    banda = handle_labels.Bf_graph(graph, solution_f)
    maior = {}
    menor ={}
    mid = {}

    for v in graph.V():
        for u in graph.N(v):
            maior[v] = solution_f[u]
            menor[v] = solution_f[u]
        mid[v] = (maior[v] + menor[v]) // 2

    
    swaps_to_consider = defaultdict(list)

    for i in graph.V():
        if max(abs(maior[i] - solution_f[i]), abs(menor[i] - solution_f[i])) != banda:
            continue
        for x in graph.N(i):
            if abs(mid[i] - solution_f[x]) < abs(mid[i] - solution_f[i]):
                swaps_to_consider[i].append((abs(mid[i] - solution_f[x]), x))
    
    for i in range(N):
        swaps_to_consider[i].sort()

    can_improve = True

    while can_improve:
        can_improve = False
        
        for i in graph.V():
            if max(abs(maior[i] - solution_f[i]), abs(menor[i] - solution_f[i])) != banda:
                continue
            for z in swaps_to_consider[i]:
                x = z[1]
                copia = solution_f

                copia[x], copia[i] = copia[i], copia[x]

                Bf_copia = handle_labels.Bf_graph(graph, copia)
                Bf_sol = handle_labels.Bf_graph(graph, solution_f)
                Vc_copia = handle_labels.Vc(graph, copia)
                Vc_sol = handle_labels.Vc(graph, solution_f)
                if Bf_copia < Bf_sol or (Bf_copia == Bf_sol and Vc_copia < Vc_sol):
                    s = copia
                    can_improve = True
                    maior = [0] * N
                    menor = [1e9] * N
                    
                    for i in graph.V():
                        for x in graph.N(i):
                            maior[i] = max(maior[i], s[x])
                            menor[i] = min(menor[i], s[x])
                    
                    mid = [(maior[i] + menor[i]) // 2 for i in graph.V()]
                    
                    banda = handle_labels.Bf_graph(graph, s)
                    
                    swaps_to_consider = defaultdict(list)
                    
                    for i in range(N):
                        if max(abs(maior[i] - s[i]), abs(menor[i] - s[i])) != banda:
                            continue
                        for x in graph.N(i):
                            if abs(mid[i] - s[x]) < abs(mid[i] - s[i]):
                                swaps_to_consider[i].append((abs(mid[i] - s[x]), x))
                    
                    for i in range(N):
                        swaps_to_consider[i].sort()
                    
                    solution_f = s 
                    break
    return solution_f

