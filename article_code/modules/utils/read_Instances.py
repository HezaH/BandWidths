import math
import sys
from collections import defaultdict
from scipy.sparse import lil_matrix, coo_matrix


def _parse_mtx_dimensions(parts):
    """Parse MatrixMarket size line.

    Common formats:
    - coordinate: nrows ncols nnz
    - array:      nrows ncols
    """
    if len(parts) >= 3:
        nrows, ncols, nnz = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
        return nrows, ncols, nnz
    if len(parts) == 2:
        nrows, ncols = (int(float(parts[0])), int(float(parts[1])))
        return nrows, ncols, 0
    raise ValueError(f"Invalid MatrixMarket size line: {' '.join(parts)}")


def _parse_mtx_edge(parts):
    """Parse an edge line using only the first two columns.

    Weighted MatrixMarket files may have 3 columns: i j value.
    We intentionally ignore extra columns to build an unweighted graph.
    """
    if len(parts) < 2:
        raise ValueError("Invalid edge line (expected at least 2 columns)")
    e1 = int(float(parts[0]))
    e2 = int(float(parts[1]))
    return e1, e2

def load_instance_fast(filename):
    neighbours = defaultdict(list)
    edges = []

    with open(filename, 'r') as f:
        # lê primeira linha útil
        for line in f:
            parts = line.split()
            if parts and not parts[0].startswith('%'):
                nnodes, value, nedges = _parse_mtx_dimensions(parts)
                break

        # lê arestas
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('%'):
                continue

            e1, e2 = _parse_mtx_edge(parts)

            neighbours[e1].append(e2)

            if e1 != e2:
                edges.append((min(e1, e2), max(e1, e2)))
                neighbours[e1].append(e2)
                neighbours[e2].append(e1)
            else:
                nedges -= 1

    # cria lista de adjacência
    nodes = sorted(neighbours.keys())
    lista_adj = [neighbours[n] for n in nodes]

    # cria matriz esparsa diretamente com COO (muito mais rápido)
    row = []
    col = []
    for u, adj in neighbours.items():
        for v in adj:
            row.append(u - 1)
            col.append(v - 1)

    matrix = coo_matrix(( [1]*len(row), (row, col) ), shape=(nnodes, nnodes))

    return nnodes, nedges, edges, neighbours, lista_adj, matrix

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
                nnodes, value, nedges = _parse_mtx_dimensions(parts)
                flag = False
            else:
                # arestas podem ter 2+ colunas (ex: i j w); usa apenas i e j
                e1, e2 = _parse_mtx_edge(parts)
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
    # ler as instancias termicas (compat): agora tolera 3 colunas e comentarios.
    edges = []
    neighbours = defaultdict(list)
    neigh = defaultdict(list)
    flag = True

    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('%'):
                continue

            if flag:
                nnodes, value, nedges = _parse_mtx_dimensions(parts)
                flag = False
                continue

            e1, e2 = _parse_mtx_edge(parts)
            neigh[e1].append(e2)
            if e1 != e2:
                edges.append((min(e1, e2), max(e1, e2)))
                neighbours[e1].append(e2)
                neighbours[e2].append(e1)
            else:
                nedges -= 1

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
