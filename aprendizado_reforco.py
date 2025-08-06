import os
import random
from collections import OrderedDict
from scipy.io import mmread
import numpy as np
from copy import deepcopy
import time
import math

def multi_centralities_multi_inicios(A, centrality_array, max_iter):
    def _getLCR(C, centrality, alpha):
        """
        Seleciona elementos i de C tal que:
        h_min <= c_i <= h_max + alpha * (h_min - h_max)
        """
        if not C:
            return []

        scores = [centrality(i) for i in C]
        h_min = min(scores)
        h_max = max(scores)
        threshold = h_max + alpha * (h_min - h_max) 

        print(f"Scores: {scores}"
              f"\nH_min: {h_min}, H_max: {h_max}, Threshold: {threshold}")
        
        L = [(i, c) for i, c in zip(C, scores) if h_min <= c <= threshold]
        
        return sorted(L, key=lambda t: t[1], reverse=True)

    def _is_connected(A):
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

    def _multi_centralities_LCR(A, centralities, alpha):
        """
        Implementa o Algoritmo Construtivo de Multi-Centralidades LCR.
        
        A            : scipy.sparse CSR matrix de adjacência (n×n)
        centralities : lista de vetores de centralidade ou único vetor
        alpha        : float em [0,1], parâmetro de aleatorização do LCR

        Retorna f: lista de tamanho n, onde f[i] é a ordem em que i foi selecionado.
        """
        n = A.shape[0]

        # 1) Constrói lista de adjacência
        adj = [A.indices[A.indptr[i]:A.indptr[i+1]].tolist() for i in range(n)]
        
        # 2) Define função de centralidade
        centrality = lambda x: centralities[x]

        # 3) Inicializações
        mark = [False] * n   # vértices já inseridos em f
        f = [0] * n       # vetor de ordem final
        l = 0             # contador de vértices ordenados

        # 4) Vértice inicial
        k = random.randrange(n)
        mark[k] = True
        l = 1
        f[k] = 1
        q = [k]        # fronteira atual

        # 5) Loop principal
        while l < n:
            # 5.1) Coleta vizinhos não marcados de todos os vértices em q
            s = []
            for i in q:
                for j in adj[i]:
                    if not mark[j]:
                        s.append(j)

            # elimina duplicatas preservando a ordem de descoberta
            s = list(OrderedDict.fromkeys(s))
            if not s:
                break

            # 5.2) Seleciona via LCR
            q_new = _getLCR(s, centrality, alpha)
            ql =  [ql[0] for ql in q_new]

            # 5.3) Marca e rotula cada vértice selecionado
            for j in ql:
                if not mark[j]:
                    l += 1
                    f[j] = l
                    mark[j] = True

            # 5.4) Atualiza fronteira
            q = s

        return f
    
    def _dcg_score(f, centrality_vector):
        ordered_centralities = [0] * len(f)
        for node, order in enumerate(f):
            ordered_centralities[order - 1] = centrality_vector[node]

        return sum(rel / np.log2(i + 2) for i, rel in enumerate(ordered_centralities))

    # verifica conectividade
    if _is_connected(A):
        print("O grafo é totalmente conectado.")
        
        # #! Cabe essa tratativa?
        # copy_A = deepcopy(A)
        # adj = [copy_A.indices[copy_A.indptr[i]:copy_A.indptr[i+1]].tolist() for i in range(copy_A.shape[0])]
        # copy_adj = deepcopy(adj)
        # for i, v in enumerate(copy_adj):
        #     if i in v:
        #         copy_adj[i].remove(i)

        best_f     = None
        best_score = np.inf
        timings    = []

        for i in range(max_iter):
            # central_vec = centrality_array[i % len(centrality_array)]

            start = time.time()
            f = _multi_centralities_LCR(A, centrality_array, alpha)
            elapsed = time.time() - start
            timings.append(elapsed)
            
            if len(set(f)) != A.shape[0]:
                raise ValueError("f contém elementos duplicados ou está incompleto!")

            # avalia com DCG
            score = _dcg_score(f, centrality_array)

            if score < best_score:
                best_score = score
                best_f = f.copy()
            print(f"Iteração {i + 1}: Ordem de seleção f = {f}")
        
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
        f = multi_centralities_multi_inicios(A, centrality_array, max_iter=10)
        print("Ordem de seleção f:\n", f)

    a  = 1