import numpy as np
import random
from ..utils import handle_labels 
from ..graph.Grafo import Grafo, GrafoListaAdj
import time

def control(graph: GrafoListaAdj, solution_f:dict, k:int, k_min:int, k_step:int)->dict:
    k_l_max = 100
    solution_return = None
    if k<=k_l_max:
        solution_return = shake_one(graph, solution_f, k)
        return solution_return
    else:
        k_l = int((k-k_min)/k_step)
        solution_return = shake_two(graph, solution_f=solution_f, k=k_l)
        return solution_return

def shake_one(graph: GrafoListaAdj, solution_f:dict, k:int)->dict:
    solution_band = {}
    
    solution_f_return = None

    for v in graph.V():
        solution_band[v] = ((solution_f[v], handle_labels.Bf_node(graph, solution_f, v)))
    
    list_all_bands = [tupla[1] for tupla in solution_band.values()]
    list_all_bands = list(set(list_all_bands))
    list_all_bands_sorted = sorted(list_all_bands, reverse=True)
    nodes_sorted = sorted(solution_band.items(), key=lambda item: item[1][1], reverse=True)
    # print("nodes_sorted: ", nodes_sorted)
    top_k_nodes = []
    
    for band in list_all_bands_sorted:
        for node in nodes_sorted:
            if node[1][1] == band:
                top_k_nodes.append(node)
            elif len(top_k_nodes)>=k:
                break
    
    
    for i in range(k):
        u_tuple = random.choice(top_k_nodes)
        u = u_tuple[0] # vertice u
        # print("u: ", u_tuple)  
        considers_v = []
        bandwidth = handle_labels.Bf_graph(graph, solution_f)

        

        vertex_neighboors = []
        for vi in graph.N(u):
            vertex_neighboors.append((vi, solution_f[vi]))
            if (abs(solution_f[u]-solution_f[vi]) == bandwidth):
                considers_v.append(v)

        if len(considers_v) > 0:
            v = random.choice(considers_v)
            f_v = solution_f[v]
            # print("i: ", i)
            # print("f_v: ", f_v)
        else:
            continue

        # print("vertex_neighboors: ", vertex_neighboors)
        tuple_fu_max = max(vertex_neighboors, key=lambda tuple: tuple[1])
        tuple_fu_min = min(vertex_neighboors, key=lambda tuple: tuple[1])
        fu_max = tuple_fu_max[1]
        fu_min = tuple_fu_min[1]

        possibles_w = []
        for node_w in nodes_sorted:
            w = node_w[0]
            f_w = node_w[1][0]
            # band_w = node_w[1][1]
            # print("fu_min: ", fu_min)
            # print("fu_max: ", fu_max)
            # print("f_w: ", f_w)
            
            if fu_min <= f_w and f_w <= fu_max:
                #vertices amostras de fmin(w) e fmax(w)
                vertexs_fw = []
                for vi in graph.N(w):
                    vertexs_fw.append((vi, solution_f[vi]))

                tuple_fw_max = max(vertexs_fw, key=lambda tuple: tuple[1])
                tuple_fw_min = min(vertexs_fw, key=lambda tuple: tuple[1])
                fw_max = tuple_fw_max[1]
                fw_min = tuple_fw_min[1]

                max_value = max(abs(fw_max-f_v), abs(f_v-fw_min))
                possibles_w.append((w, max_value))
        
        if len(possibles_w) > 0:
            tuple_w_arg_min = min(possibles_w, key=lambda tuple: tuple[1])
            w_arg_min = tuple_w_arg_min[0]
            solution_f_return = handle_labels.swapping(solution_f, v, w_arg_min)
        else:
            continue
    if solution_f_return == None:
        return solution_f
    else:
        return solution_f_return

def shake_two(graph: GrafoListaAdj, solution_f:dict, k:int)->dict:
    '''
    Funcionando profundamente, agradecimentos ao Balthazar Paixão
    '''
    bandwidth = handle_labels.Bf_graph(graph, solution_f)
    n = graph.n
    solution_result = solution_f.copy()
    # print("n ",n)
    for i in range(k):
        copia = solution_f.copy()
        begin = random.randint(1, n)
        # print("begin: ", begin)
        # print("band: ", bandwidth)
        end = begin + random.randint(1, min(20, bandwidth))
        # print("end: ", end)
        # middle = random.randint(1, end - begin - 1)
        
        if end > n:
            end = n

        for j in range(begin, end+1):
            if j < end:
                solution_result[j] = copia[j+1]
            else:
                solution_result[j] = copia[begin]
    
    return solution_result
