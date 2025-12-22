import math
import sys
from collections import defaultdict

from collections import defaultdict

def load_instance(filename):
    edges = []
    neighbours = defaultdict(list)
    neigh = defaultdict(list)
    flag = True
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()

            if flag:
                # primeira linha: cabeçalho
                nnodes, value, nedges = map(int, parts)
                flag = False
            else:
                # converte todos os termos para int
                nums = [int(float(x)) for x in parts]

                # exemplo: se a linha tem pelo menos 2 termos
                e1, e2 = nums[0], nums[1]
                neigh[e1].append(e2)

                if e1 != e2:
                    edges.append((min(e1, e2), max(e1, e2)))
                    neighbours[e1].append(e2)
                    neighbours[e2].append(e1)
                else:
                    nedges -= 1

                # se houver mais termos além de e1 e e2, você pode tratar aqui:
                if len(nums) > 2:
                    extra = nums[2:]
                    # faça algo com os extras, se necessário

    f = []
    
    for v in neighbours:
        f.append(v)

    nodes = []

    for v in neighbours:
        nodes.append(v)

    lista_adj = []

    for n in nodes:
        lista_adj.append(neighbours[n])

    
    matrix = []

    for key in neighbours:
        line = [0]*nnodes
        for element in neighbours[key]:
            line[element-1] = 1
        matrix.append(line)
    
    return nnodes, nedges, edges, neighbours, lista_adj, matrix

def load_instance_x(filename):
    #ler as instancias thermais
    edges = []
    neighbours = defaultdict(list)
    neigh = defaultdict(list)
    flag = True
    f = open(filename, 'r')
    for line in f:

        if flag == True:
            nnodes, value, nedges = [int(x) for x in line.split()]
            flag = False
        else:
            e1, e2 = [x for x in line.split()]
            e1 = int(e1)
            e2 = int(e2)
            neigh[e1].append(e2)
            if e1 != e2:
                edges.append((min(e1, e2), max(e1, e2)))
                neighbours[e1].append(e2)
                neighbours[e2].append(e1)
            else:
                nedges = nedges - 1
    f.close()

    f = []
    
    for v in neighbours:
        f.append(v)

    nodes = []

    for v in neighbours:
        nodes.append(v)

    lista_adj = []

    for n in nodes:
        lista_adj.append(neighbours[n])

    
    matrix = []

    for key in neighbours:
        line = [0]*nnodes
        for element in neighbours[key]:
            line[element-1] = 1
        matrix.append(line)
    
    return nnodes, nedges, edges, neighbours, lista_adj, matrix

def print_instance(qtd_nodes, qtd_edges, edges, neighbours):
    print(str(qtd_nodes)+" "+str(qtd_edges))

    for e in edges:
        print(str(e[0])+" "+str(e[1]))

    for i in neighbours:
        print(neighbours[i])

    print(neighbours)
