import os
import random
import time
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


# -------------------------------
# Funções utilitárias
# -------------------------------
def plot_graph(A, arquivo):
    """Plota padrão de conexões de uma matriz esparsa."""
    if isinstance(A, nx.Graph):
        A = nx.to_scipy_sparse_array(A, dtype=int)
        
    plt.figure(figsize=(6, 6))
    plt.spy(A, markersize=1)
    plt.title(f"Matriz de Adjacência {arquivo}")
    plt.savefig(f"figures/{arquivo}.png", format="png")
    plt.close()


def calc_bandwidth(A, order):
    """Calcula largura de banda do grafo."""
    pos = {node: idx for idx, node in enumerate(order)}
    return max(abs(pos[u] - pos[v]) for u, v in A.edges)


def get_LCR(candidates, centrality_vector, alpha):
    """Constrói Lista de Candidatos Restrita (LCR) para um nível da BFS."""
    if not candidates:
        return []
    # Ordena os vizinhos com base na centralidade escolhida
    sorted_candidates = sorted(candidates, key=lambda x: centrality_vector[x])
    h_min = centrality_vector[sorted_candidates[0]]
    h_max = centrality_vector[sorted_candidates[-1]]

    threshold = h_max + alpha * (h_min - h_max)
    lcr = [v for v in sorted_candidates if centrality_vector[v] <= threshold]
    # Embaralha a LCR para introduzir aleatoriedade
    random.shuffle(lcr)
    return lcr


def bfs_LCR(A, centralities_list, alpha, maxiter=10):
    """Executa BFS com seleção via LCR e retorna a melhor ordem encontrada."""
    best_order, best_bandwidth, best_elapsed = None, float('inf'), None

    for _ in range(maxiter):
        visited, order = set(), []

        while len(visited) < len(A):
            start_node = random.choice([v for v in A.nodes if v not in visited])
            visited.add(start_node)
            order.append(start_node)

            queue = deque([start_node])
            while queue:
                current_level = list(queue)
                queue.clear()

                centrality_vector = random.choice(centralities_list)

                neighbors = {nb for node in current_level for nb in A.neighbors(node) if nb not in visited}
                lcr = get_LCR(list(neighbors), centrality_vector, alpha)

                for v in lcr:
                    visited.add(v)
                    order.append(v)
                    queue.append(v)

        start_time = time.perf_counter()
        bandwidth = calc_bandwidth(A, order)
        elapsed = time.perf_counter() - start_time

        if bandwidth < best_bandwidth:
            best_bandwidth, best_order, best_elapsed = bandwidth, order, elapsed

    return best_order, best_bandwidth, best_elapsed


def is_connected(A):
    """Verifica se um grafo (matriz esparsa CSR) é conexo."""
    adj = [A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(A.shape[0])]
    visited = [False] * len(adj)
    queue = [0]
    visited[0] = True

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    return all(visited)


def read_unweighted_graph(file_path):
    """Lê arquivo .mtx como grafo não ponderado."""
    A = mmread(file_path).tocsr()
    A.data[:] = 1
    return nx.from_scipy_sparse_array(A)


def reorder_graph(A, order):
    """Reordena matriz de adjacência ignorando pesos."""
    if isinstance(A, nx.Graph):
        A = nx.to_scipy_sparse_array(A, dtype=np.int8, weight=None)
    A = csr_matrix(A)
    A.data[:] = 1
    order = np.array(order)
    return A[order, :][:, order]


# -------------------------------
# Função principal
# -------------------------------
def main_function(file_path, arquivo):
    A = read_unweighted_graph(file_path)

    # Converte para CSR para verificação rápida
    A_csr = nx.to_scipy_sparse_array(A, dtype=np.int8)
    if not is_connected(A_csr):
        raise ValueError(f"Grafo {arquivo} não é conexo.")

    print(f"O grafo {arquivo} é totalmente conectado.")

    # Calcula centralidades
    timings = {}
    centralities = {}
    for name, func in {
        "degree": nx.degree_centrality,
        "closeness": nx.closeness_centrality,
        "betweenness": nx.betweenness_centrality
    }.items():
        start = time.perf_counter()
        centralities[name] = func(A)
        timings[name] = time.perf_counter() - start
        print(f"Cálculo da centralidade {name} levou {timings[name]:.4f} s.")

    centralities_list = [[centralities[name][v] for v in A.nodes] for name in centralities]
    alpha = random.uniform(0, 1)

    order, bw, bt = bfs_LCR(A, centralities_list, alpha=alpha, maxiter=20)
    print("Melhor ordem encontrada:", order)
    print("Largura de banda:", bw)
    print(f"Tempo decorrido: {int(bt // 60)} min {int(bt % 60)} s.")

    A_new = reorder_graph(A, order)
    plot_graph(A_new, arquivo)


# -------------------------------
# Execução
# -------------------------------
if __name__ == "__main__":
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix')
    mtx_files = [f for f in os.listdir(log_folder) if f.endswith('.mtx')]

    for arquivo in mtx_files:
        print(f"Arquivo .mtx encontrado: {arquivo}")
        try:
            file_path = os.path.join(log_folder, arquivo)
            main_function(file_path, arquivo)
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
