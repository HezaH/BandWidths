"""
==================================================================================
fast_centralities + benchmark_centralities

Calcula métricas de centralidade sobre grafos, mas primeiro TESTA o custo real
de cada uma contra o PageRank (a referência de "barato e informativo" usada no
artigo original) e descarta automaticamente as que forem custosas demais --
sem você precisar saber de antemão qual é leve e qual não é.

Fluxo recomendado:
  1. Pegue 1 (ou poucos) grafos REPRESENTATIVOS do seu dataset (ex.: o maior
     grafo, ou um grafo de tamanho médio).
  2. Rode benchmark_centralities() nele -> ele mede o tempo de cada métrica
     e devolve só as que custam até `max_ratio` vezes o tempo do PageRank.
  3. Use o dict filtrado (leve) para processar TODOS os grafos do dataset
     com fast_centralities(), com segurança de que nada vai travar o script.

Particularidades tratadas (ver explicação completa nos comentários das
funções):
  - Current-flow Betweenness: exige grafo conexo e quebra em componentes
    com menos de 3 nós -> calculado componente a componente.
  - Eigenvector / Katz Centrality: podem não convergir -> tentativa extra
    com mais iterações antes de desistir.
  - Qualquer centralidade: se falhar mesmo assim, vira 0.0 e NÃO derruba
    o cálculo das demais.
  - Benchmark: a primeira chamada de qualquer função paga o custo de
    "lazy import" de bibliotecas internas (scipy etc.), o que infla o
    tempo medido -- por isso há uma rodada de aquecimento (warm-up) antes
    da medição real.
==================================================================================
"""

import time
import numpy as np
import networkx as nx


# --------------------------------------------------------------------------------
# 1) Cálculo robusto de uma centralidade (trata particularidades conhecidas)
# --------------------------------------------------------------------------------
def _safe_call_centrality(name, func, G):
    """Roteia cada centralidade para o tratamento adequado às suas particularidades."""

    if name == "Current-flow Betweenness":
        return _per_connected_component(G, func)

    if name in ("Eigenvector", "Katz Centrality"):
        try:
            return func(G, max_iter=1000)
        except TypeError:
            return func(G)
        except nx.PowerIterationFailedConvergence:
            return func(G, max_iter=2000, tol=1e-4)

    return func(G)


def _per_connected_component(G, func):
    """
    Aplica uma métrica que só é definida para grafos conexos, componente a
    componente. Componentes com menos de 3 nós recebem 0.0 diretamente
    (current-flow betweenness não é definida ali -- divisão por zero).
    """
    values = {}
    for component_nodes in nx.connected_components(G):
        if len(component_nodes) < 3:
            for n in component_nodes:
                values[n] = 0.0
            continue
        subG = G.subgraph(component_nodes).copy()
        values.update(func(subG))
    return values


# --------------------------------------------------------------------------------
# 2) Benchmark: mede o custo real de cada centralidade e filtra as leves
# --------------------------------------------------------------------------------
def benchmark_centralities(G, dict_centralities, reference="PageRank", max_ratio=3.0, verbose=True):
    """
    Mede o tempo de cada centralidade num grafo de amostra e mantém apenas
    as que custam até `max_ratio` vezes o tempo da referência (PageRank
    por padrão -- a métrica mais barata e informativa do artigo original).

    G                 : networkx.Graph OU matriz de adjacência (numpy array)
    dict_centralities : dict {"nome": funcao_networkx}
    reference         : nome da métrica usada como referência de "barato"
    max_ratio         : quantas vezes mais lenta que a referência ainda é aceitável
    verbose           : imprime a tabela de tempos e o veredito de cada métrica

    Retorna: dict_centralities FILTRADO, só com as métricas consideradas leves.
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        G = nx.from_numpy_array(np.asarray(G))

    # aquecimento: descarta o custo de "lazy import" que infla a 1a chamada
    for name, func in dict_centralities.items():
        try:
            _safe_call_centrality(name, func, G)
        except Exception:
            pass

    times = {}
    for name, func in dict_centralities.items():
        t0 = time.time()
        try:
            _safe_call_centrality(name, func, G)
            times[name] = time.time() - t0
        except Exception:
            times[name] = float("inf")  # falhou -> tratada como "infinitamente cara"

    ref_time = times.get(reference)
    if not ref_time or ref_time <= 0 or ref_time == float("inf"):
        ref_time = min([t for t in times.values() if 0 < t < float("inf")], default=1e-6)

    leves = {}
    if verbose:
        print(f"Referência ({reference}): {ref_time:.5f}s | limite: {max_ratio}x")
    for name, t in sorted(times.items(), key=lambda kv: kv[1]):
        ratio = t / ref_time if ref_time > 0 else float("inf")
        mantida = ratio <= max_ratio
        if mantida:
            leves[name] = dict_centralities[name]
        if verbose:
            status = "MANTIDA" if mantida else "descartada"
            print(f"  {name:28s}: {t:.5f}s  ({ratio:6.1f}x) -> {status}")

    return leves


# --------------------------------------------------------------------------------
# 3) Aplicação em lote: usa o dict já filtrado para processar qualquer grafo
# --------------------------------------------------------------------------------
def fast_centralities(G, dict_centralities, verbose=False):
    """
    Calcula as centralidades de `dict_centralities` (idealmente já filtrado
    por benchmark_centralities) para o grafo G.

    Retorna: dict {"nome_da_metrica": {node: valor}}
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        G = nx.from_numpy_array(np.asarray(G))

    centralities_list = {}
    for name, func in dict_centralities.items():
        t0 = time.time()
        try:
            centralities_list[name] = _safe_call_centrality(name, func, G)
            if verbose:
                print(f"  [{name:28s}] ok em {time.time() - t0:.4f}s")
        except Exception as e:
            centralities_list[name] = {n: 0.0 for n in G.nodes()}
            if verbose:
                print(f"  [{name:28s}] FALHOU ({type(e).__name__}: {e}) -> preenchido com 0.0")
    return centralities_list