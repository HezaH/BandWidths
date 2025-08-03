import os
import random
from collections import OrderedDict
from scipy.io import mmread
import numpy as np
from copy import deepcopy

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
        Implementa o Algoritmo Construtivo de Multi-Centralidades LCR
        A: matriz de adjacência esparsa (CSR) de dimensão n×n
        centralities: lista de arrays de centralidade (um por vértice) ou um único array
        alpha: parâmetro de aleatorização do LCR
        Retorna f, um vetor de ordem de inserção dos vértices (1..n)
        """
        n = A.shape[0]
        # 1) Constrói lista de adjacência
        adj = [A.indices[A.indptr[i]:A.indptr[i+1]].tolist() for i in range(n)]
        # 2) Define função de centralidade
        centrality = lambda x: centralities[x]

        # 3) Inicializações
        mark = [False] * n    # vértices já ordenados
        f = [0] * n    # posição no tour

        # 4) Escolhe vértice inicial aleatoriamente
        k = random.randrange(n)
        print(f"Vértice inicial escolhido: {k}")
        mark[k] = True
        q = [k] 
        ql = 1
        f[k] = 1
        l = 1

        # 5) Loop principal até ordenar todos os vértices
        while l < n:
            # 5.1) Constrói candidatos s = vizinhos não marcados de q
            r = 0
            s = []
            for il in range(ql):
                i = q[il]
                print(f"Vertice {i}, contido na lista ql")
                for jl in adj[i]:
                    # Varre os vizinhos de i
                    j = jl
                    print(f"Verificando vizinho {j} do vertice {i}")
                    if not mark[j]:
                        print(f"Vértice {j} não marcado, adicionando a s")
                        r += 1
                        s.append(j)
                        mark[j] = True

            # elimina duplicatas mantendo ordem de descoberta
            # s = list(OrderedDict.fromkeys(s))

            # 5.2) Seleciona subconjunto via LCR
            q = _getLCR(s, centrality, alpha)
            ql =  [ql[0] for ql in q]
            j_star = 0
            
            # 5.3) Marca e rotula cada vértice selecionado
            for jl in ql: #range(ql):
                j = q[j_star][0]
                l += 1
                f[j] = l
                j_star += 1
                # if not mark[j]:
                #     l      += 1
                #     f[j]    = l
                #     mark[j] = True

            # 5.4) Atualiza fronteira para próxima iteração
            q = s
            ql = r

        return f
   
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

        f_sol = np.inf

        for i in range(max_iter):
            f = _multi_centralities_LCR(A, centrality_array, alpha)

            if sum(f) < f_sol:
                f_sol = sum(f)
            print(f"Iteração {i + 1}: Ordem de seleção f = {f}")

        return f_sol

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
