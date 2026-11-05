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



def local_search_band(graph: GrafoListaAdj, solution_f: dict, max_passes: int = 100,
                      max_candidates: int = 50, time_limit: float = None) -> dict:
    '''
    Busca local do VNS-band (Mladenovic et al., 2010), baseada em Lim et al. (2006).

    Para cada vertice critico v (|f(v) - f(w)| = banda para algum vizinho w), calcula
    mid(v) = (menor rotulo vizinho + maior rotulo vizinho) // 2 e tenta trocar o rotulo de v
    com o de um vertice u cujo rotulo esteja mais proximo de mid(v) do que f(v)
    (candidatos ordenados pela distancia a mid(v), no maximo max_candidates).
    A troca e aceita se nenhuma aresta passa a exceder a banda e o numero de arestas
    criticas diminui; quando chega a zero, a banda caiu e o processo continua com a nova banda.

    Avaliacao incremental: cada troca so recalcula as arestas incidentes a v e u
    (as funcoes hill_climbing / improved_hill_climbing acima recalculam a banda inteira
    a cada candidato e trocam rotulos sobre a solucao original).

    Retorna uma nova solucao (dicionario vertice -> rotulo); solution_f nao e alterada.
    '''
    from ..components.constructive import get_neighbors_cache

    start_time = time.time()
    neighbors = get_neighbors_cache(graph)
    n = graph.n
    f = [0] * (n + 1)      # vertice -> rotulo
    inv = [0] * (n + 1)    # rotulo -> vertice
    for v in graph.V():
        f[v] = solution_f[v]
        inv[f[v]] = v

    edges_u, edges_v = handle_labels.get_edges_cache(graph)
    if len(edges_u) == 0:
        return dict(solution_f)

    def global_band():
        labels = np.asarray(f)
        diffs = np.abs(labels[edges_u + 1] - labels[edges_v + 1])
        band = int(diffs.max())
        critical = diffs == band
        critical_nodes = set((edges_u[critical] + 1).tolist()) | set((edges_v[critical] + 1).tolist())
        return band, int(critical.sum()), critical_nodes

    def swap_delta(v, u, band):
        '''(viola, criticas_antes, criticas_depois) considerando so as arestas de v e u.'''
        fv, fu = f[v], f[u]
        before = after = 0
        for w in neighbors[v]:
            before += abs(fv - f[w]) == band
            d = abs(fu - (fv if w == u else f[w]))
            if d > band:
                return True, 0, 0
            after += d == band
        for w in neighbors[u]:
            if w == v:
                continue  # aresta uv ja contada
            before += abs(fu - f[w]) == band
            d = abs(fv - f[w])
            if d > band:
                return True, 0, 0
            after += d == band
        return False, before, after

    band, n_critical, critical_nodes = global_band()
    for _ in range(max_passes):
        improved = False
        for v in list(critical_nodes):
            if time_limit is not None and time.time() - start_time > time_limit:
                return {v: f[v] for v in graph.V()}
            labels_nb = [f[w] for w in neighbors[v]]
            if max(abs(f[v] - l) for l in labels_nb) != band:
                continue  # deixou de ser critico apos trocas anteriores
            mid = (min(labels_nb) + max(labels_nb)) // 2
            dist_v = abs(mid - f[v])
            # rotulos mais proximos de mid(v) do que f(v), do mais proximo para o mais distante
            candidates = []
            for d in range(dist_v):
                for label in (mid - d, mid + d) if d else (mid,):
                    if 1 <= label <= n and inv[label] != v:
                        candidates.append(inv[label])
                if len(candidates) >= max_candidates:
                    break
            for u in candidates[:max_candidates]:
                violates, before, after = swap_delta(v, u, band)
                if violates or after >= before:
                    continue
                f[v], f[u] = f[u], f[v]
                inv[f[v]], inv[f[u]] = v, u
                n_critical += after - before
                improved = True
                break
            if n_critical <= 0:
                break  # banda caiu: recalcula antes de continuar
        if n_critical <= 0 or improved:
            # banda caiu (ou o conjunto de criticos mudou): recalcula o estado global
            band, n_critical, critical_nodes = global_band()
        if not improved:
            break

    return {v: f[v] for v in graph.V()}
