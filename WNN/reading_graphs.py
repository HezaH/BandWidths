"""
==================================================================================
Leitor de grafos (formato TUDataset) + extração das métricas de centralidade
citadas no artigo "Updating KernelCanvas for weightless graph classification"
(grau, PageRank, onion layer / k-core number)

VERSÃO OTIMIZADA: paraleliza o cálculo das métricas entre os grafos usando
todos os núcleos disponíveis do processador (ProcessPoolExecutor), já que o
cálculo de cada grafo é totalmente independente dos demais. Também escreve o
CSV de forma incremental, em vez de acumular tudo em memória antes de salvar.

Como usar:
  1. Ajuste BASE_DIR abaixo para o caminho da sua pasta "data_sets".
  2. Ajuste a lista `datasets_to_process` no final do arquivo.
  3. Rode: python main.py

Estrutura esperada de cada dataset (padrão TUDataset), dentro de
BASE_DIR/<NOME_DATASET>/:
    <NOME>_A.txt                -> lista de arestas (obrigatório)
    <NOME>_graph_indicator.txt  -> a qual grafo cada nó pertence (obrigatório)
    <NOME>_graph_labels.txt     -> rótulo de cada grafo (opcional)
    <NOME>_node_labels.txt      -> rótulo categórico de cada nó (opcional)
    <NOME>_node_attributes.txt  -> atributos contínuos de cada nó (opcional)

Saída: um arquivo CSV "<NOME>_centrality_metrics.csv" salvo dentro da própria
pasta do dataset, com uma linha por nó, contendo grau, PageRank e onion layer.
==================================================================================
"""

import os
import csv
import time
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

try:
    from .centralities import benchmark_centralities, fast_centralities
except ImportError:
    from centralities import benchmark_centralities, fast_centralities


# --------------------------------------------------------------------------------
# 0) CONFIGURAÇÃO — ajuste este caminho para a sua pasta local
# --------------------------------------------------------------------------------
BASE_DIR = os.path.join("WNN", "data_sets")

# Nº de processos paralelos. None = usa todos os núcleos disponíveis (os.cpu_count()).
# Se seu PC travar ou esquentar demais, reduza esse número (ex.: N_WORKERS = 4).
N_WORKERS = None

# Métricas candidatas que passam pelo benchmark antes de entrarem no worker.
# PageRank continua como referência do custo.
BENCHMARK_CENTRALITIES = {
    "Closeness": nx.closeness_centrality,
    "Betweenness": nx.betweenness_centrality,
    "Eigenvector": nx.eigenvector_centrality,
    "Katz Centrality": nx.katz_centrality,
    "Harmonic Centrality": nx.harmonic_centrality,
    "Current-flow Betweenness": nx.current_flow_betweenness_centrality,
    "K-core number": nx.core_number,
    "Onion layer": nx.onion_layers,
}

REFERENCE_CENTRALITY = "PageRank"
DEFAULT_MAX_RATIO = 3.0


# Estado compartilhado pelos workers do pool.
_WORKER_EXTRA_CENTRALITIES = {}


def _init_worker_centralities(extra_centralities):
    global _WORKER_EXTRA_CENTRALITIES
    _WORKER_EXTRA_CENTRALITIES = extra_centralities or {}


# --------------------------------------------------------------------------------
# 1) Funções auxiliares de leitura dos arquivos TUDataset
# --------------------------------------------------------------------------------
def read_lines_int(path):
    with open(path, "r") as f:
        return [int(x.strip()) for x in f if x.strip()]


def read_lines_floats(path):
    """Lê linhas com um ou mais valores separados por vírgula. Dividindo e convertendo string em numero"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])
    return rows


def load_tu_dataset(dataset_dir, dataset_name):
    """
    Lê um dataset no formato TUDataset e retorna:
      graphs        -> lista de networkx.Graph, um por grafo do dataset
      graph_labels  -> lista de rótulos (um por grafo), ou None se não existir
    """
    # Monta os caminhos esperados dos arquivos padrão do TUDataset.
    # Alguns são obrigatórios (A e graph_indicator) e outros opcionais.
    path_A = os.path.join(dataset_dir, f"{dataset_name}_A.txt")
    path_indicator = os.path.join(dataset_dir, f"{dataset_name}_graph_indicator.txt")
    path_node_labels = os.path.join(dataset_dir, f"{dataset_name}_node_labels.txt")
    path_node_attrs = os.path.join(dataset_dir, f"{dataset_name}_node_attributes.txt")
    path_graph_labels = os.path.join(dataset_dir, f"{dataset_name}_graph_labels.txt")

    # Valida a presença dos arquivos mínimos para reconstruir os grafos:
    # - A.txt: lista de arestas (nós em indexação global)
    # - graph_indicator.txt: mapeia cada nó global para qual grafo ele pertence
    if not os.path.exists(path_A) or not os.path.exists(path_indicator):
        raise FileNotFoundError(
            f"Não encontrei '{dataset_name}_A.txt' e/ou "
            f"'{dataset_name}_graph_indicator.txt' em: {dataset_dir}"
        )

    # Lê, para cada nó global, o id do grafo ao qual ele pertence (1..n_graphs).
    graph_indicator = read_lines_int(path_indicator)
    n_graphs = max(graph_indicator)

    # Inicializa uma lista com um grafo NetworkX por id de grafo do dataset.
    graphs = [nx.Graph() for _ in range(n_graphs)]

    # Constrói o mapeamento de nó global -> (índice do grafo, índice local do nó).
    # Isso permite converter os ids globais dos arquivos TU para ids locais em cada grafo.
    node_map = {}
    local_counters = [0] * n_graphs
    for global_id, g_id in enumerate(graph_indicator, start=1):
        g_idx = g_id - 1
        local_id = local_counters[g_idx]
        local_counters[g_idx] += 1
        node_map[global_id] = (g_idx, local_id)
        graphs[g_idx].add_node(local_id)

    # Se existir, anexa rótulo categórico de nó em graphs[g].nodes[n]["label"].
    if os.path.exists(path_node_labels):
        node_labels = read_lines_int(path_node_labels)
        for global_id, label in enumerate(node_labels, start=1):
            g_idx, local_id = node_map[global_id]
            graphs[g_idx].nodes[local_id]["label"] = label

    # Se existir, anexa atributos numéricos de nó em graphs[g].nodes[n]["attr"].
    # Mantém escalar quando há apenas 1 valor; caso contrário, salva lista.
    if os.path.exists(path_node_attrs):
        node_attrs = read_lines_floats(path_node_attrs)
        for global_id, attrs in enumerate(node_attrs, start=1):
            g_idx, local_id = node_map[global_id]
            graphs[g_idx].nodes[local_id]["attr"] = attrs[0] if len(attrs) == 1 else attrs

    # Lê arestas em indexação global, converte para indexação local e adiciona ao grafo correto.
    # Se uma aresta ligar nós de grafos diferentes (caso inconsistente), ela é ignorada.
    with open(path_A, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a_str, b_str = line.split(",")
            a, b = int(a_str), int(b_str)
            g_idx_a, local_a = node_map[a]
            g_idx_b, local_b = node_map[b]
            if g_idx_a != g_idx_b:
                continue
            graphs[g_idx_a].add_edge(local_a, local_b)

    # Rótulos de grafo são opcionais (ex.: tarefas de classificação de grafos).
    graph_labels = None
    if os.path.exists(path_graph_labels):
        graph_labels = read_lines_int(path_graph_labels)

    # Retorna a coleção de grafos reconstruídos e, quando disponível, seus rótulos.
    return graphs, graph_labels


def plot_graph(
    graphs,
    graph_index,
    graph_labels=None,
    layout_seed=42,
    node_size=500,
    with_labels=True,
    show=True,
    save_path=None,
    figsize=(10, 8),
):
    """Representa visualmente um grafo da lista retornada por ``load_tu_dataset``.

    Parameters
    ----------
    graphs : list[networkx.Graph]
        Lista de grafos do dataset.
    graph_index : int
        Índice baseado em zero do grafo que será desenhado.
    graph_labels : list, optional
        Rótulos dos grafos, usados no título quando fornecidos.
    layout_seed : int, optional
        Semente do layout para gerar uma disposição reproduzível.
    node_size : int, optional
        Tamanho dos nós na figura.
    with_labels : bool, optional
        Se ``True``, exibe o índice de cada nó.
    show : bool, optional
        Se ``True``, chama ``matplotlib.pyplot.show()``.
    save_path : str, optional
        Caminho onde a figura será salva.
    figsize : tuple, optional
        Tamanho da figura em polegadas.

    Returns
    -------
    tuple
        ``(fig, ax)`` do Matplotlib, para permitir customizações posteriores.
    """
    if not isinstance(graph_index, int):
        raise TypeError("graph_index deve ser um inteiro baseado em zero")
    if graph_index < 0 or graph_index >= len(graphs):
        raise IndexError(
            f"graph_index={graph_index} fora do intervalo válido "
            f"[0, {len(graphs) - 1}]"
        )

    graph = graphs[graph_index]
    if not isinstance(graph, nx.Graph):
        raise TypeError("O item selecionado não é um objeto networkx.Graph")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    positions = nx.spring_layout(graph, seed=layout_seed)
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=ax,
        with_labels=with_labels,
        node_color="#4C78A8",
        edge_color="#777777",
        node_size=node_size,
        font_color="white",
        font_weight="bold",
    )

    title = f"Grafo {graph_index} | nós: {graph.number_of_nodes()} | arestas: {graph.number_of_edges()}"
    if graph_labels is not None:
        if len(graph_labels) != len(graphs):
            raise ValueError("graph_labels deve ter o mesmo tamanho de graphs")
        title += f" | rótulo: {graph_labels[graph_index]}"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


# --------------------------------------------------------------------------------
# 2) Extração das métricas de centralidade (Seção 2.1.1) — roda em cada worker
# --------------------------------------------------------------------------------
def _compute_metrics_worker(args):
    """
    Roda em cada processo paralelo. Recebe (g_idx, G, label) e devolve:
      - rows      : uma linha por NÓ (centralidades + node_label + node_attr_*)
      - edge_rows : uma linha por ARESTA (edge_label + edge_attr_*), só quando existirem
    """
    g_idx, G, label = args

    degree = dict(G.degree())
    pagerank = nx.pagerank(G, tol=1e-4) if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}
    # onion = nx.onion_layers(G)
    # k_core = nx.core_number(G)

    extra_metrics = (
        fast_centralities(G, _WORKER_EXTRA_CENTRALITIES, verbose=False)
        if _WORKER_EXTRA_CENTRALITIES else {}
    )

    # ---- NOVO: coleta node_label / node_attr (podem não existir no dataset) ----
    node_label_raw = {n: G.nodes[n].get("label") for n in G.nodes()}
    has_node_label = any(v is not None for v in node_label_raw.values())

    node_attr_raw = {n: G.nodes[n].get("attr") for n in G.nodes()}
    has_node_attr = any(v is not None for v in node_attr_raw.values())

    rows = []
    for node in G.nodes():
        row = {
            "graph_id": g_idx,
            "graph_label": label,
            "node_id": node,
            "degree": degree[node],
            "pagerank": round(pagerank[node], 6),
            # "onion_layer": onion[node],
            # "k_core": k_core[node],
            **{
                metric_name: round(metric_values[node], 6)
                for metric_name, metric_values in extra_metrics.items()
            },
        }
        if has_node_label:
            row["node_label"] = node_label_raw[node]
        if has_node_attr:
            val = node_attr_raw[node]
            if isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    row[f"node_attr_{i}"] = round(v, 6)
            else:
                row["node_attr_0"] = round(val, 6)
        rows.append(row)

    # ---- NOVO: coleta edge_label / edge_attr (uma linha por aresta) ----
    edge_label_raw = {e: G.edges[e].get("label") for e in G.edges()}
    has_edge_label = any(v is not None for v in edge_label_raw.values())

    edge_attr_raw = {e: G.edges[e].get("attr") for e in G.edges()}
    has_edge_attr = any(v is not None for v in edge_attr_raw.values())

    edge_rows = []
    if has_edge_label or has_edge_attr:
        for e in G.edges():
            erow = {"graph_id": g_idx}
            if has_edge_label:
                erow["edge_label"] = edge_label_raw[e]
            if has_edge_attr:
                val = edge_attr_raw[e]
                if isinstance(val, (list, tuple)):
                    for i, v in enumerate(val):
                        erow[f"edge_attr_{i}"] = round(v, 6)
                else:
                    erow["edge_attr_0"] = round(val, 6)
            edge_rows.append(erow)

    return g_idx, G.number_of_nodes(), G.number_of_edges(), rows, edge_rows


# --------------------------------------------------------------------------------
# 3) Processamento de um dataset inteiro, em paralelo, com escrita incremental
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 4) PARTE 1: extração em paralelo + escrita incremental dos CSVs (nós e arestas)
# --------------------------------------------------------------------------------
def process_dataset(dataset_name, base_dir=BASE_DIR, save_csv=True, n_workers=N_WORKERS,
                     extra_centralities=None, verbose=True):
    """Processa um dataset TUDataset e salva as métricas calculadas em CSV.

        Etapas principais:
            1. Localiza e carrega os grafos do dataset.
            2. Cria tarefas independentes para processamento paralelo.
            3. Calcula métricas de nós e arestas em múltiplos processos.
            4. Escreve as linhas dos resultados incrementalmente nos CSVs.
            5. Fecha os arquivos, remove uma saída de arestas vazia e retorna os caminhos.
        """

    # 1) Localiza a pasta do dataset e carrega os grafos e seus rótulos.
    dataset_dir = os.path.join(base_dir, dataset_name)
    print(f"\n=== [M] Extraindo métricas: {dataset_name} ===")

    t0 = time.time()
    graphs, graph_labels = load_tu_dataset(dataset_dir, dataset_name)
    # plot_graph(graphs, graph_index=5, graph_labels=graph_labels)
    n_graphs = len(graphs)
    print(f"Total de grafos encontrados: {n_graphs} (leitura em {time.time() - t0:.1f}s)")

    # 2) Monta uma tarefa por grafo. O índice e o rótulo acompanham o grafo
    # para que os workers possam identificar cada resultado corretamente.
    tasks = [
        (g_idx, G, (graph_labels[g_idx] if graph_labels else "N/A"))
        for g_idx, G in enumerate(graphs)
    ]

    # 3) Define os arquivos de saída: um CSV por nó e outro por aresta.
    out_path_nodes = os.path.join(dataset_dir, f"{dataset_name}_centrality_metrics.csv")
    out_path_edges = os.path.join(dataset_dir, f"{dataset_name}_edge_metrics.csv")

    writer_nodes = writer_edges = None
    csv_file_nodes = open(out_path_nodes, "w", newline="") if save_csv else None
    csv_file_edges = open(out_path_edges, "w", newline="") if save_csv else None

    # Contadores usados para acompanhar o progresso e gerar um resumo final.
    n_nodes_total = n_edges_total = processed = 0
    any_edge_rows = False
    t1 = time.time()

    # 4) Distribui os grafos entre os processos. O initializer configura em
    # cada worker as centralidades extras que serão usadas no cálculo.
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker_centralities,
        initargs=(extra_centralities,),
    ) as executor:
        chunksize = max(1, n_graphs // ((n_workers or os.cpu_count() or 1) * 4) or 1)
        for g_idx, n_nodes, n_edges, rows, edge_rows in executor.map(
            _compute_metrics_worker, tasks, chunksize=chunksize
        ):
            # Acumula estatísticas gerais do processamento.
            n_nodes_total += n_nodes
            n_edges_total += n_edges
            processed += 1

            # 5) Cria o cabeçalho na primeira resposta e grava as métricas de nós
            # sem manter todos os resultados na memória.
            if save_csv and rows:
                if writer_nodes is None:
                    writer_nodes = csv.DictWriter(csv_file_nodes, fieldnames=rows[0].keys())
                    writer_nodes.writeheader()
                writer_nodes.writerows(rows)

            # Grava métricas de arestas apenas quando o grafo possui atributos
            # de aresta. A flag evita manter um CSV vazio ao final.
            if save_csv and edge_rows:
                any_edge_rows = True
                if writer_edges is None:
                    writer_edges = csv.DictWriter(csv_file_edges, fieldnames=edge_rows[0].keys())
                    writer_edges.writeheader()
                writer_edges.writerows(edge_rows)

            if verbose and processed % max(1, n_graphs // 10) == 0:
                print(f"  ... {processed}/{n_graphs} grafos processados")

    # 6) Fecha os arquivos para garantir que todo o conteúdo seja persistido.
    if csv_file_nodes:
        csv_file_nodes.close()
    if csv_file_edges:
        csv_file_edges.close()

    # Se nenhum grafo tinha atributos de aresta, remove o CSV criado sem dados.
    if save_csv and not any_edge_rows and os.path.exists(out_path_edges):
        os.remove(out_path_edges)
        out_path_edges = None

    # 7) Exibe um resumo do processamento e retorna os caminhos das saídas.
    elapsed = time.time() - t1
    print(f"Total de nós: {n_nodes_total} | Total de arestas: {n_edges_total}")
    if elapsed > 0:
        print(f"Tempo de cálculo: {elapsed:.1f}s ({n_graphs / elapsed:.1f} grafos/s)")
    if save_csv:
        print(f"Métricas de nó salvas em: {out_path_nodes}")
        print(f"Métricas de aresta salvas em: {out_path_edges}" if out_path_edges else
              "Nenhum atributo de aresta encontrado neste dataset (sem CSV de arestas)")

    return out_path_nodes, out_path_edges
