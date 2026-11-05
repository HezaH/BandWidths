#regular imports
import numpy as np
import random
import time
import networkx as nx
import heapq
#import models
from modules.utils import handle_labels
from modules.graph.Grafo import Grafo, GrafoListaAdj

def get_LCR(queue:list, chosen_centrality:dict, random_centrality:str, alpha:float, centralities:dict):
    """
    Constructs a Restricted Candidate List (RCL) by sorting vertices based on centrality 
    and partitioning them into restricted and complementary sets.
    This function implements the RCL creation step commonly used in GRASP (Greedy Randomized 
    Adaptive Search Procedure) algorithms. It sorts vertices according to a chosen centrality 
    metric, selects the top alpha% as restricted candidates, and randomly shuffles both 
    the restricted and complementary sets before combining them.
    Args:
        queue (list): List of vertex identifiers to be processed.
        chosen_centrality (dict): Dictionary mapping each vertex to its centrality value.
        random_centrality (str): Key identifying which centrality metric is being used.
        alpha (float): Parameter between 0 and 1 that determines the proportion of top-ranked 
                      vertices to include in the restricted candidate list (0.0 to 1.0).
        centralities (dict): Dictionary containing centrality metric configurations, including 
                           a "reverse" boolean flag indicating sort order for the chosen metric.
    Returns:
        list: A shuffled list of vertices starting with randomized high-centrality vertices 
              (restricted set) followed by randomized remaining vertices (complementary set).
    """
    reverse_flag = centralities[random_centrality]["reverse"]
    qtd_elem = int(len(queue) * alpha)

    # pega top k
    if reverse_flag:
        LCR = heapq.nlargest(qtd_elem, queue, key=lambda x: chosen_centrality[x])
    else:
        LCR = heapq.nsmallest(qtd_elem, queue, key=lambda x: chosen_centrality[x])

    # marca os escolhidos
    mark = {v: True for v in LCR}

    # complementares sem custo extra
    LCR_comp = [v for v in queue if v not in mark]

    random.shuffle(LCR)
    random.shuffle(LCR_comp)

    return LCR + LCR_comp

def init_Solution(graph: GrafoListaAdj)->dict:
    ''''FUNCIONANDO PROFUNDAMENTE'''
    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 #O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #O primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0 #
    
    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        h = random.randint(0, ql-1)
        
        for j1 in range(ql):
            n = queue[h]
            l = l + 1
            f[n] = l
            h = h + 1
            if h >= ql:
                h = 0
        
        queue = stack
        ql = r
        
    return f


def init_Solution_Centrality(graph: GrafoListaAdj, edges:list, heuristic)->dict:
    
    # calculando a centralidade de cada vertice
    if heuristic != None:
        G = nx.Graph()
        G.add_edges_from(edges)
        resp_centrality = heuristic[0](G)
        nodes_centrality = {}
        for node, centrality in resp_centrality.items():
            nodes_centrality[node] = centrality
    

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        
        
        vertices_ordenados = sorted(queue, key=lambda x: nodes_centrality[x], reverse=heuristic[1])
        
        h = 0
        for j1 in range(ql):
            n = vertices_ordenados[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f

def init_Solution_Centrality_otm(graph: GrafoListaAdj, nodes_centrality:dict)->dict:
    
    # calculando a centralidade de cada vertice

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    random_centrality = nodes_centrality 

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        vertices_ordenados = sorted(queue, key=lambda x: random_centrality[x], reverse=True)
        
        
        h = 0
        for j1 in range(ql):
            n = vertices_ordenados[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f

def get_neighbors_cache(graph: GrafoListaAdj) -> list:
    """
    Lista de adjacência (índice = vértice) construída uma única vez por grafo.

    graph.N(v) é um gerador que percorre a lista encadeada a cada chamada; em um
    multi-start com centenas de construções isso domina o custo. O cache fica
    guardado no próprio objeto e é invalidado se o número de arestas mudar.
    """
    cache = getattr(graph, "_neighbors_cache", None)
    if cache is None or cache[0] != graph.m:
        neighbors = [[]] + [list(graph.N(v)) for v in graph.V()]
        cache = (graph.m, neighbors)
        graph._neighbors_cache = cache
    return cache[1]


# Estratégias para escolher o vértice inicial (rótulo 1) de cada construção
START_STRATEGIES = (
    "centrality",          # sorteio proporcional à centralidade (comportamento original do Q-MCH)
    "inverse_centrality",  # sorteio proporcional a 1/centralidade (favorece vértices periféricos)
    "uniform",             # sorteio uniforme (como o construtivo do VNS-band)
    "peripheral",          # vértice pseudo-periférico (George-Liu), como RCM/GPS (survey, Seção 3.2)
)


def get_start_weights(graph: GrafoListaAdj, nodes_centrality: dict, random_centrality: str,
                      start_strategy: str = "centrality") -> tuple:
    """
    Vértices e pesos para o sorteio do vértice inicial.
    Calculado uma vez por (grafo, centralidade, estratégia) em vez de a cada construção.
    """
    cache = getattr(graph, "_start_weights_cache", None)
    if cache is None or cache[0] != graph.m:
        cache = (graph.m, {})
        graph._start_weights_cache = cache
    key = (random_centrality, start_strategy)
    if key not in cache[1]:
        V = list(graph.V())
        # pesos alinhados com V (vértice sem centralidade vira 0)
        values = [nodes_centrality.get(v, 0) for v in V]
        if start_strategy == "inverse_centrality":
            positive = [x for x in values if x > 0]
            eps = min(positive) / 2 if positive else 1.0
            values = [1.0 / (x + eps) for x in values]
        elif start_strategy != "centrality":
            values = [0]
        # random.choices aceita pesos não normalizados; soma zero -> pesos uniformes
        cache[1][key] = (V, values if sum(values) > 0 else None)
    return cache[1][key]


def _bfs_levels(neighbors: list, root: int) -> list:
    """Níveis da busca em largura a partir de root (lista de listas de vértices)."""
    seen = {root}
    frontier = [root]
    levels = [frontier]
    while frontier:
        next_level = []
        for u in frontier:
            for w in neighbors[u]:
                if w not in seen:
                    seen.add(w)
                    next_level.append(w)
        if next_level:
            levels.append(next_level)
        frontier = next_level
    return levels


def pseudo_peripheral_node(neighbors: list, start: int) -> int:
    """Algoritmo de George-Liu (1979): parte de `start` e salta para o vértice de menor grau
    do último nível enquanto a excentricidade (nº de níveis) aumentar."""
    v = start
    levels = _bfs_levels(neighbors, v)
    while True:
        last = min(levels[-1], key=lambda x: len(neighbors[x]))
        new_levels = _bfs_levels(neighbors, last)
        if len(new_levels) <= len(levels):
            return v
        v, levels = last, new_levels


def get_peripheral_pool(graph: GrafoListaAdj, n_starts: int = 16) -> list:
    """Conjunto de vértices pseudo-periféricos obtidos a partir de n_starts vértices aleatórios.
    Calculado uma vez por grafo; cada construção sorteia um deles (mantém a diversidade
    do multi-start sem pagar várias BFS por construção)."""
    cache = getattr(graph, "_peripheral_pool_cache", None)
    if cache is None or cache[0] != graph.m:
        neighbors = get_neighbors_cache(graph)
        V = list(graph.V())
        pool = sorted({pseudo_peripheral_node(neighbors, random.choice(V)) for _ in range(n_starts)})
        cache = (graph.m, pool)
        graph._peripheral_pool_cache = cache
    return cache[1]


def choose_start_vertex(graph: GrafoListaAdj, nodes_centrality: dict, random_centrality: str,
                        start_strategy: str = "centrality") -> int:
    """Escolhe o vértice inicial segundo `start_strategy` (ver START_STRATEGIES)."""
    if start_strategy not in START_STRATEGIES:
        raise ValueError(f"start_strategy inválida: {start_strategy}. Use uma de {START_STRATEGIES}")
    if start_strategy == "peripheral":
        return random.choice(get_peripheral_pool(graph))
    V, weights = get_start_weights(graph, nodes_centrality, random_centrality, start_strategy)
    return random.choices(V, weights=weights)[0]


def init_Solution_Centrality_lcr(graph: GrafoListaAdj, nodes_centrality:dict,
                                 random_centrality:str, alpha:float, centralities:dict,
                                 parent_order: bool = True, start_strategy: str = "centrality")->dict:
    """
    Construtivo multi-centralidade com LCR (Algoritmo 1 do artigo SBPO).

    parent_order=True  -> regra de Cuthill-McKee: os vértices de um nível são rotulados
                          na ordem do rótulo do pai, e a LCR ordena apenas os filhos de
                          cada pai. Mantém vizinhos com rótulos próximos (survey, Seção 3.2).
    parent_order=False -> comportamento original: o nível inteiro passa pela LCR, que
                          embaralha os vértices e perde a ordem dos pais (banda ~30-45% maior).
    start_strategy     -> como escolher o vértice inicial (ver START_STRATEGIES).
    """
    V = list(graph.V())
    n = len(V)

    k = choose_start_vertex(graph, nodes_centrality, random_centrality, start_strategy)

    neighbors = get_neighbors_cache(graph)

    if parent_order:
        # busca em largura em que cada vértice é rotulado ao ser descoberto:
        # filhos de um pai com rótulo menor sempre recebem rótulos menores
        f = {k: 1}
        order = [k]
        i1 = 0
        while i1 < len(order):
            i = order[i1]
            i1 += 1
            children = [vert for vert in neighbors[i] if vert not in f]
            if children:
                array_LCR = get_LCR(queue=children, chosen_centrality=nodes_centrality,
                                    random_centrality=random_centrality, alpha=alpha,
                                    centralities=centralities)
                for vert in array_LCR:
                    order.append(vert)
                    f[vert] = len(order)
            if i1 == len(order) and len(order) < n:
                # grafo desconexo: continua a busca a partir de um vértice ainda não rotulado
                k = next(v for v in V if v not in f)
                order.append(k)
                f[k] = len(order)
        return f

    # f e mark mais rápidos
    f = {v: None for v in V}
    mark = {v: False for v in V}

    mark[k] = True
    queue = [k]

    ql = 1
    f[k] = 1
    l = 0

    while l < n:
        r = 0
        stack = []

        for i1 in range(ql):
            i = queue[i1]
            for vert in neighbors[i]:
                if not mark[vert]:
                    r += 1
                    stack.append(vert)
                    mark[vert] = True

        array_LCR = get_LCR(queue=queue, chosen_centrality=nodes_centrality,
                            random_centrality=random_centrality, alpha=alpha,
                            centralities=centralities)

        h = 0
        for j1 in range(ql):
            n2 = array_LCR[h]
            l += 1
            f[n2] = l
            h += 1

        queue = stack
        ql = r

    return f
