import random
from collections import deque
import networkx as nx
import os
import random
from collections import OrderedDict
from scipy.io import mmread
import numpy as np
from copy import deepcopy
import time
import math
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix

def plot_graph(A, arquivo):
    # ===== Visualização da matriz como imagem =====
    plt.figure(figsize=(6, 6))
    plt.spy(A, markersize=1)  # Visualiza padrão de conexões
    plt.title(f"Matriz de Adjacência {arquivo}")
    plt.savefig(f"figures/{arquivo}.png", format="png")
    return plt

def calc_bandwidth(A, order):
    """
    Calcula a largura de banda de uma matriz de adjacência A.
    A largura de banda é definida como a maior diferença absoluta entre os índices
    das linhas e colunas de cada aresta.
    """
    value = max(abs(order.index(u) - order.index(v)) for u, v in A.edges)
    return value

def get_LCR(candidates, centrality_vector, alpha):
    """Constrói Lista de Candidatos Restrita (LCR) a partir dos candidatos do nível atual"""
    if not candidates:
        return []
    # Ordena candidatos pela centralidade (menor -> maior)
    sorted_candidates = sorted(candidates, key=lambda x: centrality_vector[x])
    h_min = centrality_vector[sorted_candidates[0]]
    h_max = centrality_vector[sorted_candidates[-1]]

    # Limite para inclusão na LCR
    threshold = h_max + alpha * (h_min - h_max) 
    print(f"Scores: {sorted_candidates}"
        f"\nH_min: {h_min}, H_max: {h_max}, Threshold: {threshold}")
        
    lcr = [v for v in sorted_candidates if centrality_vector[v] <= threshold]

    # Retorna LCR embaralhada (para aleatoriedade)
    random.shuffle(lcr)
    return lcr

def bfs_LCR(A, centralities_list, alpha, maxiter=10):
    """
    Implementa BFS com LCR nível a nível, variando centralidades e multi-start
    Retorna a melhor ordem de vértices encontrada
    """

    best_order = None
    best_bandwidth = float('inf')

    for _ in range(maxiter):

        start = time.perf_counter()
        visited = set()
        order = []

        # Enquanto houver vértices não visitados
        while len(visited) < len(A):
            # Escolhe aleatoriamente um vértice inicial não visitado
            start = random.choice([v for v in A.nodes if v not in visited])
            visited.add(start)
            order.append(start)

            # BFS estruturada por níveis
            queue = deque([start])
            while queue:
                current_level = list(queue)
                queue.clear()

                # Escolhe aleatoriamente uma centralidade para este nível
                centrality_vector = random.choice(centralities_list)

                # Coleta vizinhos não visitados deste nível
                neighbors = set()
                for node in current_level:
                    for nb in A.neighbors(node):
                        if nb not in visited:
                            neighbors.add(nb)

                # Aplica LCR neste conjunto
                lcr = get_LCR(list(neighbors), centrality_vector, alpha)

                # Rotula na ordem definida pela LCR
                for v in lcr:
                    visited.add(v)
                    order.append(v)
                    queue.append(v)

        # Calcula largura de banda (menor é melhor)
        bandwidth = calc_bandwidth(A, order)
        end = time.perf_counter()
        elapsed = end - start

        if bandwidth < best_bandwidth:
            best_bandwidth = bandwidth
            best_order = order
            best_elapsed = elapsed

    return best_order, best_bandwidth, best_elapsed

def is_connected(A):
        """
        Verifica se o grafo é totalmente conectado (conexo) usando BFS.
        Entrada: adj -> lista de adjacência.
        Saída: True se for conexo, False caso contrário.
        """
        adj = [A.indices[A.indptr[i]:A.indptr[i+1]].tolist() for i in range(A.shape[0])]

        n = len(adj)
        visited = [False] * n
        queue = [0]  # começa do vértice 0
        visited[0] = True

        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        return all(visited)

def read_unweighted_graph(file_path):
    # Lê matriz esparsa
    A = mmread(file_path).tocsr()

    # Remove pesos (torna todas as arestas = 1)
    A.data[:] = 1

    # Converte para grafo não ponderado
    A = nx.from_scipy_sparse_array(A)  # padrão: sem peso
    A = nx.to_scipy_sparse_array(A, dtype=np.int8)
    return A

def reorder_graph(A, order):
    """Reordena grafo/matriz esparsa ignorando pesos"""
    # Se A for grafo NetworkX, converte para matriz esparsa sem pesos
    if isinstance(A, nx.Graph):
        A = nx.to_scipy_sparse_array(A, dtype=np.int8, weight=None)  # ignora pesos

    # Garante CSR para indexação eficiente
    A = csr_matrix(A)

    # Força todas as arestas a valerem 1
    A.data[:] = 1

    # Aplica a reordenação
    order = np.array(order)
    return A[order, :][:, order]

# ============================
# Exemplo de uso
# ============================
if __name__ == "__main__":
    # Caminho da pasta 'matrix' relativa a este script
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix')

    # Lista apenas os arquivos que terminam com .mtx
    mtx_files = [f for f in os.listdir(log_folder) 
                if os.path.isfile(os.path.join(log_folder, f)) and f.endswith('.mtx')]

    # Percorre e exibe cada arquivo .mtx
    for arquivo in mtx_files:
        # try:
            print("Arquivo .mtx encontrado:", arquivo)

            file_path = os.path.join(log_folder, arquivo)

            # Leitura e conversão da matriz
            A = read_unweighted_graph(file_path=file_path)
            
            if is_connected(A):
                print(f"O grafo {arquivo} é totalmente conectado.")

                A = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.Graph())
                # Lista de centralidades (uma por métrica)
                start = time.perf_counter()
                degree_centrality = nx.degree_centrality(A)
                end = time.perf_counter()
                print(f"Cálculo da centralidade de grau levou {end - start:.4f} segundos.")

                start = time.perf_counter()
                closeness_centrality = nx.closeness_centrality(A)
                end = time.perf_counter()
                print(f"Cálculo da centralidade de proximidade levou {end - start:.4f} segundos.")
                
                start = time.perf_counter()
                betweenness_centrality = nx.betweenness_centrality(A)
                end = time.perf_counter()
                print(f"Cálculo da centralidade de intermediação levou {end - start:.4f} segundos.")

                centralities_list = [
                   degree_centrality,
                   closeness_centrality,
                   betweenness_centrality
                ]

                # Converte dicionários para listas indexadas por vértice
                centralities_list = [
                    [centrality[v] for v in A.nodes] for centrality in centralities_list
                ]

                # Alpha aleatório
                alpha = random.uniform(0, 1)

                order, bw, bt = bfs_LCR(A, centralities_list, alpha=alpha, maxiter=20)
                print("Melhor ordem encontrada:", order)
                print("Largura de banda:", bw)
                minutes, seconds = divmod(int(bt), 60)
                print("Tempo decorrido:", minutes, "minutos e", seconds, "segundos.")
                A_new = reorder_graph(A, order)

                plot = plot_graph(A_new, arquivo)
            else:
                print(f"O grafo {arquivo} não é totalmente conectado.")
                raise ValueError(f"Grafo {arquivo} não é conexo.")
        # except:
        #     continue