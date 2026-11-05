### Regular Import
import time

### Import to run the graph lib
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

### Import from own-code
from modules.utils.handle_labels import *
from modules.utils.read_Instances import *
from modules.utils.read_filenames import *
from modules.graph.Grafo import GrafoListaAdj
from modules.centralities.heuristics import *
from modules.components import constructive
from modules.VNS.local_search import local_search_band

def get_centrality_node(graph: nx.Graph, centrality: dict) -> dict:
    return dict(centrality["func"](graph, **centrality["args"]))

def closeness_centrality_aprox(graph: nx.Graph, k: int = 256, seed: int = 0, chunk: int = 32) -> dict:
    """
    Closeness aproximada por amostragem de k vértices-fonte (Eppstein & Wang, 2004).

    nx.closeness_centrality faz uma BFS em Python a partir de TODOS os vértices (O(n·m)),
    o que inviabiliza grafos com dezenas de milhares de vértices. Aqui as BFS rodam em C
    (scipy.sparse.csgraph) a partir de k fontes, em blocos de `chunk` fontes para limitar
    a memória (chunk x n distâncias). Com k >= n o resultado é exato e igual ao do networkx
    para grafos conexos: (n-1) / soma das distâncias.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return {v: 0.0 for v in nodes}
    matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, format="csr")
    rng = np.random.default_rng(seed)
    sources = np.arange(n) if k is None or k >= n else rng.choice(n, size=k, replace=False)

    dist_sum = np.zeros(n)
    for start in range(0, len(sources), chunk):
        dist = shortest_path(matrix, directed=False, unweighted=True, indices=sources[start:start + chunk])
        dist[np.isinf(dist)] = 0  # vértices inalcançáveis não contribuem (mesma convenção do networkx)
        dist_sum += dist.sum(axis=0)

    if len(sources) < n:
        # amostra: extrapola a soma das distâncias para os n-1 outros vértices
        dist_sum *= (n - 1) / len(sources)
    with np.errstate(divide="ignore"):
        closeness = np.where(dist_sum > 0, (n - 1) / dist_sum, 0.0)
    return dict(zip(nodes, closeness.tolist()))

def centrality_heuristic(graph:GrafoListaAdj, centrality_values:dict, cent_str:str, alpha:float, iter_max:int, centralities:dict, rebuild_graph:bool=False, parent_order:bool=True, start_strategy:str="centrality", local_search:bool=False)->int:
    """
    Executa iter_max construções com a centralidade cent_str e devolve a melhor.

    start_strategy -> escolha do vértice inicial (constructive.START_STRATEGIES).
    local_search   -> aplica a busca local do VNS-band (local_search_band) a cada construção.
                      Cada linha de `it` registra a banda da construção ("construction_bandwidth")
                      e a banda final ("bandwidth", após a busca local quando ativa).
    rebuild_graph=False evita reconstruir o grafo reetiquetado (O(m) em listas encadeadas)
    a cada melhoria; graph_rebuilt é devolvido como None nesse caso. A solução devolvida
    é sempre relativa aos vértices de `graph`.
    """

    bandwidth = float("inf")
    solution = None
    best_solution = None
    graph_rebuilt = None
    it = []
    for _ in range(iter_max):
        start_time = time.time()
        solution = constructive.init_Solution_Centrality_lcr(graph=graph, nodes_centrality=centrality_values, random_centrality=cent_str, alpha=alpha, centralities=centralities, parent_order=parent_order, start_strategy=start_strategy)

        band_solution = Bf_graph(graph=graph, F_labels=solution)
        construction_bandwidth = band_solution
        construction_time = time.time() - start_time

        if local_search:
            solution = local_search_band(graph, solution)
            band_solution = Bf_graph(graph=graph, F_labels=solution)

        it.append({
            "bandwidth": band_solution,
            "construction_bandwidth": construction_bandwidth,
            "centrality": cent_str,
            "alpha": alpha,
            "start_strategy": start_strategy,
            "local_time": time.time() - start_time,
            "construction_time": construction_time,
        })

        if bandwidth > band_solution:
            best_solution = solution
            bandwidth = band_solution

    if rebuild_graph and best_solution is not None:
        graph_rebuilt = reconstruct_graph_by_labels(graph, best_solution)

    return bandwidth, graph_rebuilt, best_solution, it
