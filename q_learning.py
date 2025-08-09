import random
import numpy as np
import time
from pyqlearning.qlearning import QLearning

from collections import OrderedDict
import os
import random
from scipy.io import mmread
import numpy as np
from copy import deepcopy
import time
import math


def multi_centralities_qlearning(A, centrality_array, max_iter):
    def _is_connected(A):
        adj = [A.indices[A.indptr[i]:A.indptr[i+1]].tolist() for i in range(A.shape[0])]
        n = len(adj)
        visited = [False] * n
        queue = [0]
        visited[0] = True

        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        return all(visited)

    def _dcg_score(f, centrality_vector):
        ordered_centralities = [0] * len(f)
        for node, order in enumerate(f):
            ordered_centralities[order - 1] = centrality_vector[node]

        return sum(rel / np.log2(i + 2) for i, rel in enumerate(ordered_centralities))

    # Inicializa Q-learning
    qlearn = QLearning(
        alpha=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995
    )

    if _is_connected(A):
        print("O grafo é totalmente conectado.")

        best_f = None
        best_score = -np.inf
        timings = []

        n = A.shape[0]
        adj = [A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(n)]

        for episode in range(max_iter):
            start = time.time()

            state = "init"
            visited = [False] * n
            f = [0] * n
            l = 1

            # Seleciona nó inicial aleatoriamente
            current_node = random.randrange(n)
            visited[current_node] = True
            f[current_node] = l
            l += 1
            q = [current_node]

            while l <= n:
                # Coleta vizinhos não visitados
                neighbors = []
                for node in q:
                    for v in adj[node]:
                        if not visited[v]:
                            neighbors.append(v)
                candidates = list(OrderedDict.fromkeys(neighbors))

                if not candidates:
                    break

                # Estado atual é a string do vetor de seleção atual
                state = str(f)

                # Escolhe a próxima ação (nó) com base no Q-learning
                action = qlearn.select_action(state, candidates)

                # Marca e atualiza
                if not visited[action]:
                    f[action] = l
                    visited[action] = True
                    l += 1

                # Próxima fronteira
                q = [action]

                next_state = str(f)

                # Recompensa final quando vetor estiver completo
                if l > n:
                    reward = _dcg_score(f, centrality_array)
                else:
                    reward = 0

                qlearn.update_q_table(state, action, reward, next_state)

            elapsed = time.time() - start
            timings.append(elapsed)

            # Avaliação
            score = _dcg_score(f, centrality_array)

            if score > best_score:
                best_score = score
                best_f = f.copy()

            print(f"Episódio {episode + 1}: f = {f}, DCG = {score:.4f}, epsilon = {qlearn.get_epsilon():.4f}")

        print(f"\nMelhor DCG geral: {best_score:.4f}")
        return best_f, timings

    else:
        print("O grafo NÃO é totalmente conectado.")
        return None

if __name__ == "__main__":
    # Caminho da pasta 'matrix' relativa a este script
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix')

    # Lista apenas os arquivos que terminam com .mtx
    mtx_files = [f for f in os.listdir(log_folder) 
                if os.path.isfile(os.path.join(log_folder, f)) and f.endswith('.mtx')]

    # Percorre e exibe cada arquivo .mtx
    for arquivo in mtx_files:
        try:
            print("Arquivo .mtx encontrado:", arquivo)

            file_path = os.path.join(log_folder, arquivo)

            # Leitura e conversão da matriz
            A = mmread(file_path).tocsr()
            print("Dimensões:", A.shape)
            print("Número de elementos não-zero:", A.nnz)

            # Centralidade de exemplo: grau (retorna os graus dos vértices)
            centrality_array = A.getnnz(axis=1).astype(float)
            dict_grau = {i: centrality_array[i] for i in range(A.shape[0])}

            # Alpha aleatório
            alpha = random.uniform(0, 1)

            # Executa o algoritmo
            f = multi_centralities_qlearning(A, centrality_array, max_iter=10)
            print("Ordem de seleção f:\n", f)
        except:
            continue

    a  = 1