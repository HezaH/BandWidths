import math
import sys
from collections import defaultdict
from scipy.sparse import lil_matrix

def load_instance(filename):
    edges = []
    neighbours = defaultdict(list)
    neigh = defaultdict(list)
    flag = True
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()

            # ignora cabeçalho e comentários
            if not parts or parts[0].startswith('%'):
                continue

            if flag:
                # primeira linha de dados (dimensões)
                nnodes, value, nedges = map(int, parts)
                flag = False
            else:
                # converte todos os termos para int
                e1, e2 = int(float(parts[0])), int(float(parts[1]))
                neigh[e1].append(e2)

                if e1 != e2:
                    edges.append((min(e1, e2), max(e1, e2)))
                    neighbours[e1].append(e2)
                    neighbours[e2].append(e1)
                else:
                    nedges -= 1

    # lista de nós
    nodes = list(neighbours.keys())

    # lista de adjacência
    lista_adj = [neighbours[n] for n in nodes]

    # matriz esparsa (muito mais leve que lista de listas)
    matrix = lil_matrix((nnodes, nnodes), dtype=int)
    for key, adj in neighbours.items():
        for element in adj:
            matrix[key-1, element-1] = 1

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
