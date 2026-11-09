# lendo todas as instancias de uma classe

if __name__ == "__main__":  # processos filhos (spawn) reimportam este arquivo
    print("[INFO] Iniciando imports do main_survey.py", flush=True)

import os
import time
import random
import argparse
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import networkx as nx
import torch
from scipy.sparse.csgraph import reverse_cuthill_mckee
from modules.utils.read_filenames import readFilesInDict
from modules.graph.Grafo import GrafoListaAdj, RedutorGrafo
from agent import Agent
from enviroment import Env
from centralities import closeness_centrality_aprox
from modules.utils import read_Instances
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.utils.handle_labels import set_bandwidth_fast
import json
import math

TIMEOUT_SECONDS = np.inf  # tempo maximo (s) para ler grafo + computar centralidades
# TIMEOUT_SECONDS = 25
# if 'C:\\Temp\\C##\\BandWidths' not in os.getcwd() :
#     TIMEOUT_SECONDS = np.inf  # desabilitar timeout se não estiver rodando no servidor

MAX_NODES_REDUCED = 2000  # tamanho do subgrafo caso estoure o timeout
REDUCTION_STRATEGY = "bfs"  # 'bfs' ou 'random'

# ------------------------------------------------------------
# PARAMETROS DO EXPERIMENTO
# ------------------------------------------------------------
MAX_ITER = 30           # episodios do agente por execucao (cada episodio = N_CALLS x ITER_MAX construcoes)
N_CALLS = 5             # chamadas de centrality_heuristic por episodio (Env.step)
ITER_MAX = 3            # construcoes por chamada
ALPHA = 0.3             # tamanho relativo da lista restrita de candidatos (LCR)
PARENT_ORDER = True     # True: niveis rotulados na ordem dos pais (Cuthill-McKee); False: construtivo original
N_RUNS = 1              # execucoes independentes por instancia (o survey recomenda >= 3)
BASE_SEED = 0           # semente da execucao r = BASE_SEED + r (reprodutibilidade)
EPS_MIN_FRACTION = 0.8  # fracao dos episodios ate epsilon chegar em eps_min (com eps_dec fixo em 0.995, 30 episodios -> eps ~ 0.86)

# ------------------------------------------------------------
# CONDICOES EXPERIMENTAIS (o que o agente escolhe a cada episodio)
# ------------------------------------------------------------
# Em cada condicao o agente varia UM parametro; os demais ficam fixos nos valores abaixo.
CONDITIONS = {
    "centrality": None,  # acoes = centralidades do dicionario `centralities` (Q-MCH original)
    "alpha": [0.0, 0.1, 0.3, 0.5, 1.0],  # acoes = alpha da LCR (0 = sem lista restrita, 1 = ordena todos pela centralidade)
    "start": ["centrality", "inverse_centrality", "uniform", "peripheral"],  # acoes = estrategia de inicio (constructive.START_STRATEGIES)
}
DEFAULT_CENTRALITY = "Degree"   # centralidade fixa quando a acao nao e a centralidade
START_STRATEGY = "centrality"   # estrategia de inicio fixa quando a acao nao e a estrategia
LOCAL_SEARCH = False            # busca local do VNS-band apos cada construcao (sobrescrito por --local-search)

# ------------------------------------------------------------
# PARALELISMO E SAIDAS
# ------------------------------------------------------------
N_WORKERS_INSTANCES = 1     # instancias processadas em paralelo (processos); 1 = sequencial
N_WORKERS_CENTRALITIES = 1  # centralidades calculadas em paralelo dentro de cada instancia
# Evite N_WORKERS_INSTANCES * N_WORKERS_CENTRALITIES > numero de nucleos.
PARALLEL_CENTRALITIES_MIN_NODES = 20000  # abaixo disso o custo de criar processos supera o ganho
SAVE_IMPROVEMENT_PLOTS = False  # True: um PNG a cada melhoria (comportamento antigo); False: apenas o melhor ao final


def _connected_graph_with_mapping(edges, nnodes):
    """Return the largest component with safe internal labels.

    MatrixMarket vertex identifiers are preserved in ``original_nodes`` and
    ``node_mapping``.  The graph used by the algorithms is relabeled to
    ``1..n`` because ``GrafoListaAdj`` stores adjacency lists by index.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(1, int(nnodes) + 1))
    graph.add_edges_from(edges)

    largest_cc = max(nx.connected_components(graph), key=len)
    original_nodes = sorted(largest_cc)
    node_mapping = {old: new for new, old in enumerate(original_nodes, start=1)}
    inverse_mapping = {new: old for old, new in node_mapping.items()}
    connected_graph = nx.relabel_nodes(
        graph.subgraph(largest_cc).copy(),
        node_mapping,
    )
    return connected_graph, node_mapping, inverse_mapping


def _save_best_graph_configuration(
    filename,
    bandwidth,
    graph,
    solution,
    inverse_mapping,
    centrality=None,
):
    """Persist the graph ordering that produced the best bandwidth."""
    reordered_graph = nx.relabel_nodes(graph, solution)
    original_labels = {
        str(inverse_mapping.get(node, node)): int(label)
        for node, label in solution.items()
    }
    configuration = {
        "bandwidth": float(bandwidth),
        "centrality": centrality,
        "nodes": int(graph.number_of_nodes()),
        "edges": [sorted(edge) for edge in reordered_graph.edges()],
        "vertex_labels": original_labels,
        "internal_vertex_labels": {
            str(node): int(label) for node, label in solution.items()
        },
        "internal_to_original": {
            str(node): int(original)
            for node, original in inverse_mapping.items()
        },
    }
    with open(filename, "w", encoding="utf-8") as configuration_file:
        json.dump(configuration, configuration_file, indent=4, sort_keys=True)


def _load_saved_bandwidth(bandwidth_path, best_graph_path):
    """Load the persisted bandwidth without reading the large configuration."""
    if os.path.exists(bandwidth_path):
        with open(bandwidth_path, "r", encoding="utf-8") as bandwidth_file:
            saved_bandwidth = json.load(bandwidth_file)
        if isinstance(saved_bandwidth, (int, float)) and math.isfinite(saved_bandwidth):
            return float(saved_bandwidth)
        raise ValueError(f"Valor inválido no arquivo de bandwidth: {bandwidth_path}")

    # Backward compatibility for configurations created before the compact file.
    if os.path.exists(best_graph_path):
        with open(best_graph_path, "r", encoding="utf-8") as configuration_file:
            configuration = json.load(configuration_file)
        saved_bandwidth = configuration.get("bandwidth")
        if isinstance(saved_bandwidth, (int, float)) and math.isfinite(saved_bandwidth):
            return float(saved_bandwidth)

    return None


def _save_bandwidth_value(filename, bandwidth):
    """Persist only the best bandwidth for fast comparisons on later runs."""
    with open(filename, "w", encoding="utf-8") as bandwidth_file:
        json.dump(float(bandwidth), bandwidth_file)


def _save_if_better(
    bandwidth,
    persisted_bandwidth,
    best_graph_path,
    bandwidth_path,
    graph,
    solution,
    inverse_mapping,
    centrality=None,
):
    """Save the full configuration and compact value only when the result improves."""
    if persisted_bandwidth is not None and bandwidth >= persisted_bandwidth:
        return persisted_bandwidth, False

    _save_best_graph_configuration(
        best_graph_path,
        bandwidth,
        graph,
        solution,
        inverse_mapping,
        centrality,
    )
    _save_bandwidth_value(bandwidth_path, bandwidth)
    return float(bandwidth), True


def configure_logger(log_file_path: str) -> logging.Logger:
    """Configure logger to write both to console and file."""
    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _compute_one_centrality(graph, centrality):
    """Calcula uma centralidade (função de topo para poder rodar em outro processo)."""
    args = dict(centrality["args"])  # cópia, para não alterar o dicionário original

    if "k" in args and args["k"] is not None and args["k"] >= len(graph):
        args["k"] = None  # grafo pequeno: calcula exato

    start = time.time()
    values = centrality["func"](graph, **args)
    return values, time.time() - start


def compute_centralities(graph, centralities, n_jobs=1):
    """Calcula todas as centralidades do grafo.

    n_jobs > 1 distribui as centralidades entre processos (as funções do networkx
    seguram o GIL, então threads não ajudam). O ganho é limitado pela centralidade
    mais lenta (em geral closeness/betweenness).

    Retorna (centralities_maps, tempos_por_centralidade).
    """
    centralities_maps, centrality_times = {}, {}
    if n_jobs <= 1 or len(centralities) <= 1 or len(graph) < PARALLEL_CENTRALITIES_MIN_NODES:
        for centrality_key, centrality in centralities.items():
            centralities_maps[centrality_key], centrality_times[centrality_key] = _compute_one_centrality(graph, centrality)
        return centralities_maps, centrality_times

    with ProcessPoolExecutor(max_workers=min(n_jobs, len(centralities)), mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_compute_one_centrality, graph, centrality): centrality_key
                   for centrality_key, centrality in centralities.items()}
        for future in as_completed(futures):
            centrality_key = futures[future]
            centralities_maps[centrality_key], centrality_times[centrality_key] = future.result()
    return centralities_maps, centrality_times


def _load_and_centralities(instancia_path, centralities, n_jobs=1):
    """Lê a instância (maior componente conexa) e calcula as centralidades."""
    try:
        nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance_fast(instancia_path)

        G_connected, node_mapping, inverse_mapping = _connected_graph_with_mapping(
            edges, nnodes
        )

        # Centralities (cache)
        centralities_maps, centrality_times = compute_centralities(G_connected, centralities, n_jobs)

        return {
            "ok": True,
            "nnodes": G_connected.number_of_nodes(),
            "edges": list(G_connected.edges()),
            "centralities_maps": centralities_maps,
            "centrality_times": centrality_times,
            "node_mapping": node_mapping,
            "inverse_mapping": inverse_mapping,
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def _worker_load_and_centralities(instancia_path, centralities, out_pipe, n_jobs=1):
    """Worker rodado em processo separado para permitir timeout (Windows-friendly)."""
    try:
        out_pipe.send(_load_and_centralities(instancia_path, centralities, n_jobs))
    finally:
        out_pipe.close()


def load_graph_and_centralities_with_timeout(instancia_path, centralities, timeout_seconds, n_jobs=1):
    """Try to load graph + compute centralities within timeout.

    Returns a dict with keys:
      - ok: bool
      - timed_out: bool
      - nnodes, edges, centralities_maps, centrality_times (when ok)
      - node_mapping, inverse_mapping (when ok)
      - error (when not ok)
    """
    if math.isinf(timeout_seconds):
        # sem timeout não há por que pagar o spawn de um processo (reimporta torch etc.)
        # e a cópia das centralidades pelo pipe
        resp = _load_and_centralities(instancia_path, centralities, n_jobs)
        resp.setdefault("timed_out", False)
        return resp

    ctx = mp.get_context("spawn")
    parent_pipe, child_pipe = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker_load_and_centralities, args=(instancia_path, centralities, child_pipe, n_jobs))
    p.start()
    child_pipe.close()

    if not parent_pipe.poll(timeout_seconds):
        if p.is_alive():
            p.terminate()
        p.join()
        parent_pipe.close()
        return {"ok": False, "timed_out": True, "error": f"timeout after {timeout_seconds}s"}
    resp = parent_pipe.recv()

    p.join()
    parent_pipe.close()
    resp.setdefault("timed_out", False)
    return resp

# Função para plotar uma matriz esparsa
def plot_sparse_matrix(matrix, title, file_name="saida.png"):
    """`matrix` pode ser densa ou scipy.sparse; use esparsa para grafos grandes."""
    plt.figure(figsize=(8, 8))
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.savefig(file_name)   # salva em arquivo
    plt.close()              # fecha a figura para não abrir


def plot_solution(G, solution, title, file_name):
    """Plota o padrão de esparsidade de G reordenado por `solution`.

    Usa matriz esparsa: nx.to_numpy_array criava uma matriz densa n x n
    (56 mil vértices -> ~25 GB), inviável nas instâncias grandes do survey.
    """
    G_reordered = nx.relabel_nodes(G, solution)
    nodes = sorted(G_reordered.nodes())
    adj_matrix_reordered = nx.to_scipy_sparse_array(G_reordered, nodelist=nodes, format="coo")
    plot_sparse_matrix(adj_matrix_reordered, title, file_name=file_name)


# ------------------------------------------------------------
# METRICAS DE REFERENCIA (extração de resultados)
# ------------------------------------------------------------

def bandwidth_of_solution(G, solution):
    """Largura de banda de G com a rotulação `solution` (vértice -> rótulo)."""
    return max((abs(solution[u] - solution[v]) for u, v in G.edges()), default=0)


def lower_bound_bandwidth(G):
    """Limite inferior da banda: máximo entre Lam et al. (1997), usado como limite dual
    no artigo SBPO, e ceil(grau_max / 2) (um vértice de grau d precisa de d rótulos vizinhos).
    O limite de Lam é próximo de zero em grafos esparsos; o de grau é o que costuma valer."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    disc = (2 * n - 1) ** 2 - 8 * m
    lb_lam = math.ceil(n - (1 + math.sqrt(disc)) / 2) if disc >= 0 else 0
    lb_degree = math.ceil(max((d for _, d in G.degree()), default=0) / 2)
    return max(lb_lam, lb_degree, 1)


def rcm_bandwidth(G):
    """Banda do Reverse Cuthill-McKee (scipy), 'estado da prática' segundo o survey.
    Custa milissegundos e serve de referência para cada instância."""
    nodes = sorted(G.nodes())
    matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")
    perm = reverse_cuthill_mckee(matrix, symmetric_mode=True)
    position = {nodes[p]: i + 1 for i, p in enumerate(perm)}
    return bandwidth_of_solution(G, position)


def seed_everything(seed):
    """Fixa as sementes usadas pelo construtivo (random), numpy e pelo DQN (torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_state(state, max_iter, nnodes):
    """Estado [passo, gap, recompensa, banda] em escalas comparáveis para a rede.
    Sem isso a banda (até dezenas de milhares) domina as entradas da DQN."""
    n_step, gap, reward, bandwidth = state
    return [n_step / max_iter, gap, reward / 10.0, bandwidth / max(nnodes, 1)]


# ------------------------------------------------------------
# ETAPAS DO FLUXO POR INSTANCIA
# ------------------------------------------------------------

def load_instance_graph(instancia, centralities, logger):
    """Lê o grafo e calcula as centralidades (com timeout e fallback para subgrafo).

    Retorna um dict com nnodes, edges, centralities_maps, node_mapping, inverse_mapping,
    reduced (bool), full_nodes e centrality_time.
    """
    logger.info("Calculando centralidades (timeout: %ss)", TIMEOUT_SECONDS)

    # 1) Tenta ler grafo + centralidades dentro do timeout
    start = time.time()
    resp = load_graph_and_centralities_with_timeout(instancia, centralities, TIMEOUT_SECONDS, N_WORKERS_CENTRALITIES)
    logger.info("Leitura e centralidades finalizadas: ok=%s timed_out=%s", resp.get("ok"), resp.get("timed_out"))

    if resp.get("ok"):
        logger.info("Tempo por centralidade (s): %s", {k: round(v, 2) for k, v in resp["centrality_times"].items()})
        return {
            "nnodes": resp["nnodes"],
            "edges": resp["edges"],
            "centralities_maps": resp["centralities_maps"],
            "node_mapping": resp["node_mapping"],
            "inverse_mapping": resp["inverse_mapping"],
            "reduced": False,
            "full_nodes": resp["nnodes"],
            "centrality_time": time.time() - start,
        }

    # 2) Fallback: reduz o grafo e recomputa centralidades no subgrafo
    if resp.get("timed_out"):
        logger.warning("[timeout] %s -> usando grafo reduzido (max_nodes=%s)", resp.get("error"), MAX_NODES_REDUCED)
    else:
        logger.warning("[erro] ao computar centralidades no grafo completo: %s -> usando grafo reduzido", resp.get("error"))

    start = time.time()
    nnodes_full, nedges_full, edges_full, neighbours, lista_adj, matrix = read_Instances.load_instance_fast(instancia)
    nnodes, edges, mapa, inv = RedutorGrafo.ReduzirArestas(
        nnodes_full,
        edges_full,
        max_nodes=MAX_NODES_REDUCED,
        estrategia=REDUCTION_STRATEGY,
        seed=0,
    )

    # NetworkX graph reduzido
    G_reduced = nx.Graph()
    G_reduced.add_edges_from(edges)
    centralities_maps, centrality_times = compute_centralities(G_reduced, centralities, N_WORKERS_CENTRALITIES)
    logger.info("Tempo por centralidade (s): %s", {k: round(v, 2) for k, v in centrality_times.items()})

    return {
        "nnodes": nnodes,
        "edges": edges,
        "centralities_maps": centralities_maps,
        "node_mapping": mapa,
        "inverse_mapping": inv,
        "reduced": True,
        "full_nodes": nnodes_full,
        "centrality_time": time.time() - start,
    }


def build_agent(n_actions, max_iter):
    """Agente DQN com decaimento de epsilon ajustado ao número de episódios.

    Com eps_dec = 0.995 e learn() chamado uma vez por episódio, 30 episódios deixam
    epsilon em ~0.86 (86% das escolhas aleatórias). Aqui eps_dec é calculado para que
    epsilon chegue a eps_min após EPS_MIN_FRACTION dos episódios.
    """
    epsilon, eps_min = 1, 0.01
    eps_dec = (eps_min / epsilon) ** (1 / max(EPS_MIN_FRACTION * max_iter, 1))
    return Agent(learning_rate=0.001,
                 gamma=0.9,
                 epsilon=epsilon,
                 eps_min=eps_min,
                 eps_dec=eps_dec,
                 n_movements=n_actions,
                 n_actions=n_actions,
                 n_states=4,
                 deep=True)


# ------------------------------------------------------------
# CONDICOES: acao do agente -> (centralidade, alpha, estrategia de inicio)
# ------------------------------------------------------------

def get_condition_actions(condition, centralities_list):
    """Lista de ações do agente na condição (ver CONDITIONS)."""
    if condition not in CONDITIONS:
        raise ValueError(f"Condição inválida: {condition}. Use uma de {list(CONDITIONS)}")
    return list(centralities_list) if CONDITIONS[condition] is None else list(CONDITIONS[condition])


def decode_action(condition, action_value):
    """Traduz o valor da ação para os parâmetros do construtivo.
    Retorna (centralidade, alpha, estratégia de início)."""
    if condition == "centrality":
        return action_value, ALPHA, START_STRATEGY
    if condition == "alpha":
        return DEFAULT_CENTRALITY, float(action_value), START_STRATEGY
    if condition == "start":
        return DEFAULT_CENTRALITY, ALPHA, action_value
    raise ValueError(f"Condição inválida: {condition}")


def condition_tag(condition, local_search):
    """Sufixo usado nos arquivos de saída, ex.: 'alpha' ou 'alpha_ls'."""
    return f"{condition}_ls" if local_search else condition


def run_agent(grafo, centralities_maps, centralities, condition, local_search, initial_bandwidth,
              lower_bound, nnodes, max_iter, logger, run_id, seed):
    """Uma execução do Q-MCH (max_iter episódios do agente) na condição `condition`.

    condition    -> o que o agente escolhe a cada episódio: "centrality", "alpha" ou "start".
    local_search -> aplica a busca local do VNS-band após cada construção.

    `grafo` nunca é substituído: todas as soluções são rotulações dos MESMOS vértices
    (os das centralidades). Antes, `grafo = grafo_solution` trocava o grafo pelo
    reetiquetado após cada melhoria, de modo que as centralidades passavam a apontar para
    vértices errados e a ordenação persistida não reproduzia a banda reportada.

    Retorna (melhor banda, melhor solução, melhor ação, linhas por construção, agente).
    """
    seed_everything(seed)
    actions = get_condition_actions(condition, list(centralities.keys()))
    agent = build_agent(len(actions), max_iter)
    env = Env(initial_bandwidth, lower_bound=lower_bound, alpha=ALPHA, n_calls=N_CALLS,
              iter_max=ITER_MAX, parent_order=PARENT_ORDER)

    state = normalize_state(env.get_initial_state(), max_iter, nnodes)
    score = 0
    temp_bandwidth = initial_bandwidth
    best_solution, best_action = None, None
    local_iteration = []

    for i in range(max_iter):
        env.reset()
        #Escolha da ação (centralidade, alpha ou estratégia de início, conforme a condição)
        action = agent.choose_action(state)
        action_value = actions[action]
        centrality, alpha, start_strategy = decode_action(condition, action_value)

        info = env.step(graph=grafo, centrality_values=centralities_maps[centrality], cent_str=centrality,
                        centralities=centralities, alpha=alpha, start_strategy=start_strategy,
                        local_search=local_search)
        reward = info["reward"]
        gap = info["gap"]
        bandwidth = info["bandwidth"]
        local_iteration.extend({**row, "Episode": i + 1, "Run": run_id, "Seed": seed,
                                "condition": condition, "action": str(action_value),
                                "local_search": local_search} for row in info["iterations"])

        new_state = normalize_state([info["n_steps"], gap, reward, bandwidth], max_iter, nnodes)

        #Treinamento do novo estado
        agent.learn(state, action, reward, new_state)

        #score acumulado por iteração
        score += reward

        logger.info(
            ">> [%s] Run %s Episode %s Gap: %.4f Reward: %s Score: %s Action: %s Bandwidth: %s Epsilon: %.3f",
            condition_tag(condition, local_search), run_id, i + 1, gap, reward, score, action_value, bandwidth, agent.epsilon,
        )
        state = new_state

        if temp_bandwidth > bandwidth:
            # salvar a melhor solução (env.best_solution corresponde a env.best_sol)
            temp_bandwidth = bandwidth
            best_solution = info["solution"].copy()
            best_action = str(action_value)
            logger.info("Melhor solução atualizada: bandwidth=%s (%s=%s)", bandwidth, condition, action_value)

    return temp_bandwidth, best_solution, best_action, local_iteration, agent


def run_condition(instancia, G, grafo, loaded, instance_info, centralities, condition, local_search, path_name, logger):
    """N_RUNS execuções de uma condição numa instância já carregada: persiste o melhor
    resultado da condição e devolve (linhas por construção, linhas de resumo)."""
    instance_path = instance_info["Instance"]
    tag = condition_tag(condition, local_search)
    best_graph_path = os.path.join(path_name, f"best_graph_{instance_path}_{tag}.json")
    bandwidth_path = os.path.join(path_name, f"best_bandwidth_{instance_path}_{tag}.json")
    initial_bandwidth = instance_info["Initial Bandwidth"]
    lower_bound = instance_info["Lower Bound"]
    rcm_bw = instance_info["RCM Bandwidth"]
    persisted_bandwidth = _load_saved_bandwidth(bandwidth_path, best_graph_path)
    persisted_bandwidth, saved = _save_if_better(
        initial_bandwidth, persisted_bandwidth, best_graph_path, bandwidth_path,
        G, {node: node for node in G.nodes()}, loaded["inverse_mapping"],
    )
    if saved:
        logger.info("[%s] Configuração inicial persistida como melhor resultado.", tag)

    df_iteration_rows, summary_rows = [], []
    best_run_bandwidth, best_agent = math.inf, None
    for run_id in range(N_RUNS):
        seed = BASE_SEED + run_id
        start_time = time.time()
        temp_bandwidth, best_solution, best_action, local_iteration, agent = run_agent(
            grafo, loaded["centralities_maps"], centralities, condition, local_search, initial_bandwidth,
            lower_bound, loaded["nnodes"], MAX_ITER, logger, run_id, seed,
        )
        global_time = time.time() - start_time

        saved = False
        if best_solution is not None:
            # confere que a ordenação a ser persistida reproduz a banda reportada
            check_bandwidth = bandwidth_of_solution(G, best_solution)
            if check_bandwidth != temp_bandwidth:
                logger.warning("Banda da solução (%s) difere da reportada (%s)", check_bandwidth, temp_bandwidth)

            persisted_bandwidth, saved = _save_if_better(
                temp_bandwidth, persisted_bandwidth, best_graph_path, bandwidth_path,
                G, best_solution, loaded["inverse_mapping"], f"{condition}={best_action}",
            )
            if saved or SAVE_IMPROVEMENT_PLOTS:
                name_matrix = f"{tag}_bandwidth_{temp_bandwidth}_{condition}={best_action}"
                file_name = os.path.join(path_name, os.path.basename(instancia).replace(".mtx", f"_{name_matrix}.png"))
                plot_solution(G, best_solution, f"{instance_path} | {name_matrix}", file_name)
                logger.info("Plot salvo: %s", file_name)
        logger.info("Melhor configuração [%s] rodada %s: bandwidth=%s; persistida=%s", tag, run_id, temp_bandwidth, saved)

        if temp_bandwidth < best_run_bandwidth:
            best_run_bandwidth, best_agent = temp_bandwidth, agent

        df_iteration = pd.DataFrame(local_iteration)
        for key, value in instance_info.items():
            df_iteration[key] = value
        df_iteration['Global Time (s)'] = global_time
        df_iteration_rows.extend(df_iteration.to_dict(orient='records'))

        # resumo por execução: média, desvio, melhor e tempo total (recomendação do survey, Seção 4.3)
        bw = df_iteration["bandwidth"]
        summary_rows.append({
            **instance_info,
            "Condition": condition,
            "Local Search": local_search,
            "Run": run_id,
            "Seed": seed,
            "Best Bandwidth": temp_bandwidth,
            "Best Action": best_action,
            "Best Construction Bandwidth": df_iteration["construction_bandwidth"].min(),
            "Mean Bandwidth": bw.mean(),
            "Std Bandwidth": bw.std(),
            "Worst Bandwidth": bw.max(),
            "Gap Lower Bound (%)": 100 * (temp_bandwidth - lower_bound) / lower_bound,
            "Gap RCM (%)": 100 * (temp_bandwidth - rcm_bw) / rcm_bw if rcm_bw else np.nan,
            "Global Time (s)": global_time,
        })

    torch_save_path = os.path.join(path_name, f'trained_model_{instance_path}_{tag}.pth')
    json_save_path = os.path.join(path_name, f'analysis_inputs_{instance_path}_{tag}.json')

    with open(json_save_path, 'w') as f:
        json.dump(df_iteration_rows, f, indent=4)
    logger.info("Analise por instancia salva em: %s", json_save_path)

    torch.save(best_agent.Q.state_dict(), torch_save_path)
    logger.info("Modelo salvo em: %s", torch_save_path)

    return df_iteration_rows, summary_rows


def process_instance(instancia, centralities, conditions=("centrality",), local_search_options=(LOCAL_SEARCH,)):
    """Fluxo completo de uma instância: leitura, centralidades (uma única vez) e, para cada
    combinação (condição, busca local), N_RUNS execuções do agente com persistência do
    melhor resultado e extração dos resultados.

    conditions           -> condições a rodar (chaves de CONDITIONS), ex.: ("alpha", "start").
    local_search_options -> (False,), (True,) ou (False, True) para comparar com/sem busca local.

    Função de topo (picklable) para poder rodar em paralelo com ProcessPoolExecutor.
    Retorna {tag: (linhas por construção, linhas de resumo)}.
    """
    if mp.current_process().name != "MainProcess":
        torch.set_num_threads(1)  # rodando dentro do pool: evita disputa de threads entre processos

    instance_path = os.path.basename(instancia).replace(".mtx", "")
    path_name = os.path.join(os.path.dirname(os.path.dirname(instancia)), "plots", f"{instance_path}")
    log_file_path = os.path.join(os.path.dirname(__file__), "logs", f"{instance_path}.log")

    # criar o diretório se não existir
    os.makedirs(path_name, exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = configure_logger(log_file_path)
    logger.info("Inicio da instancia: %s", instancia)
    logger.info("Arquivo de log salvo em: %s", log_file_path)
    logger.info("Condicoes: %s | busca local: %s", list(conditions), list(local_search_options))

    loaded = load_instance_graph(instancia, centralities, logger)
    nnodes = loaded["nnodes"]
    edges = loaded["edges"]

    # Instancia grafo NetworkX
    G = nx.Graph()
    G.add_edges_from(edges)
    nedges = G.number_of_edges()

    # Métricas de referência (extração de resultados)
    initial_bandwidth = set_bandwidth_fast(G)
    lower_bound = lower_bound_bandwidth(G)
    rcm_bw = rcm_bandwidth(G)
    logger.info("Bandwidth original: %s | RCM: %s | limite inferior: %s", initial_bandwidth, rcm_bw, lower_bound)

    # Instancia grafo da sua própria lib
    grafo = GrafoListaAdj()
    grafo.DefinirN(nnodes, VizinhancaDuplamenteLigada=True)
    for (u, v) in edges:
        grafo.AdicionarAresta(u, v)

    instance_info = {
        "Instance": instance_path,
        "Edges": nedges,
        "Nodes": nnodes,
        "Initial Bandwidth": initial_bandwidth,
        "RCM Bandwidth": rcm_bw,
        "Lower Bound": lower_bound,
        "Reduced Subgraph": loaded["reduced"],
        "Full Nodes": loaded["full_nodes"],
        "Centrality Time (s)": loaded["centrality_time"],
    }

    results = {}
    for condition in conditions:
        for local_search in local_search_options:
            results[condition_tag(condition, local_search)] = run_condition(
                instancia, G, grafo, loaded, instance_info, centralities, condition, local_search, path_name, logger,
            )
    return results


DATASETS = {"survey": "survey", "artigo": "newdata", "newdata": "newdata"}  # opção -> pasta em data/


def ask_dataset():
    """Pergunta no terminal se a execução é do survey ou do artigo; devolve a pasta em data/."""
    options = {"1": "survey", "2": "newdata"}
    print("\nQual conjunto de instancias deseja rodar?")
    print("  [1] Survey  (data/survey)")
    print("  [2] Artigo  (data/newdata - matrizes do artigo SBPO e outras)")
    while True:
        answer = input("Escolha 1 ou 2: ").strip().lower()
        if answer in options:
            return options[answer]
        if answer in DATASETS:
            return DATASETS[answer]
        print("Opcao invalida. Digite 1 (survey) ou 2 (artigo).")


def append_csv(rows, filename, first_write):
    """Grava os resultados de uma instância assim que ela termina (uma falha
    no meio da execução não perde as instâncias já concluídas)."""
    if rows:
        pd.DataFrame(rows).to_csv(filename, mode="w" if first_write else "a", header=first_write, index=False)


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # diretorio do script
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Inicio da execucao do main_survey.py ({run_timestamp})", flush=True)

    # ------------------------------------------------------------
    # CENTRALIDADES OTIMIZADAS (rápidas, com cache e versões aproximadas)
    # ------------------------------------------------------------

    centralities = {
        # Centralidades rápidas
        "Degree": {
            "func": nx.degree_centrality,
            "args": {},
            "reverse": True
        },

        # Closeness aproximada: BFS em C a partir de k fontes (exata quando k >= n)
        "Closeness": {
            "func": closeness_centrality_aprox,
            "args": {"k": 512},
            "reverse": True
        },
        # "Closeness": {
        #     "func": nx.closeness_centrality,
        #     "args": {},
        #     "reverse": True
        # },

        # "Harmonic Centrality": {
        #     "func": nx.harmonic_centrality,
        #     "args": {},
        #     "reverse": True
        # },

        # Betweenness aproximada (100× mais rápida)
        "Betweenness": {
            "func": nx.betweenness_centrality,
            "args": {"k": 50},   # amostra de nós
            "reverse": True
        },

        # Eigenvector acelerado
        "Eigenvector": {
            "func": nx.eigenvector_centrality,
            "args": {"max_iter": 200, "tol": 1e-2},
            "reverse": True
        },

        # # Katz acelerado
        # "Katz Centrality": {
        #     "func": nx.katz_centrality,
        #     "args": {"alpha": 0.005, "beta": 1.0, "max_iter": 200, "tol": 1e-2},
        #     "reverse": True
        # },

        # PageRank acelerado
        "PageRank": {
            "func": nx.pagerank,
            "args": {"alpha": 0.85, "max_iter": 100},
            "reverse": True
        }
    }

    # ------------------------------------------------------------
    # LOOP PRINCIPAL — uma tarefa por instância (sequencial ou em paralelo)
    # ------------------------------------------------------------
    # Exemplos:
    #   python main_survey.py                                         -> pergunta survey/artigo; condição "centrality", sem busca local
    #   python main_survey.py --dataset artigo                        -> não pergunta (survey | artigo | newdata)
    #   python main_survey.py --condition alpha --local-search on
    #   python main_survey.py --condition all --local-search both --dataset newdata
    #   python main_survey.py --condition start --instances bp_0 power

    parser = argparse.ArgumentParser(description="Q-MCH: redução de largura de banda com escolha de ação por DQN")
    parser.add_argument("--condition", nargs="+", default=["centrality"], choices=list(CONDITIONS) + ["all"],
                        help="o que o agente escolhe a cada episódio (pode passar mais de uma ou 'all')")
    parser.add_argument("--local-search", choices=["off", "on", "both"], default="on" if LOCAL_SEARCH else "off",
                        help="busca local do VNS-band após cada construção ('both' roda com e sem)")
    parser.add_argument("--dataset", choices=list(DATASETS), default=None,
                        help="survey = matrizes do survey; artigo/newdata = matrizes do artigo SBPO e outras. "
                             "Se omitido, a escolha é perguntada no terminal")
    parser.add_argument("--instances", nargs="*", default=None, help="restringe às instâncias com estes nomes (sem .mtx)")
    parser.add_argument("--workers", type=int, default=N_WORKERS_INSTANCES, help="instâncias processadas em paralelo")
    args = parser.parse_args()
    N_WORKERS_INSTANCES = args.workers
    args.dataset = DATASETS[args.dataset] if args.dataset else ask_dataset()
    print(f"[INFO] Dataset selecionado: {args.dataset}", flush=True)

    conditions = list(CONDITIONS) if "all" in args.condition else args.condition
    local_search_options = {"off": (False,), "on": (True,), "both": (False, True)}[args.local_search]
    tags = [condition_tag(c, ls) for c in conditions for ls in local_search_options]

    survey_file = os.path.join(base_dir, "data", args.dataset) # "survey" = dados do artigo de survey; "newdata" = dados do artigo e alguns outros novos
    filenames = {tag: os.path.join(base_dir, f"result_output_{args.dataset}_{tag}.csv") for tag in tags}
    summary_filenames = {tag: os.path.join(base_dir, f"result_summary_{args.dataset}_{tag}.csv") for tag in tags}

    dir_list = [nome for nome in os.listdir(survey_file) if os.path.isdir(os.path.join(survey_file, nome)) and not nome.startswith("plots")]
    print(f"[INFO] Total de classes em {args.dataset}: {len(dir_list)}", flush=True)

    list_path = []
    for kind in dir_list:
        path = os.path.join(survey_file, kind)
        kind_instances = readFilesInDict(path, ".mtx")
        if args.instances:
            kind_instances = [p for p in kind_instances if os.path.basename(p)[:-4] in args.instances]
        print(f"[INFO] Classe {kind} com {len(kind_instances)} instancias", flush=True)
        list_path.extend(kind_instances)

    total_rows = 0
    first_write = {tag: True for tag in tags}
    print(f"[INFO] {len(list_path)} instancias | estrategias: {tags} | {N_WORKERS_INSTANCES} processo(s) | "
          f"{N_RUNS} execucao(oes) por instancia", flush=True)

    if N_WORKERS_INSTANCES <= 1:
        results = (process_instance(instancia, centralities, conditions, local_search_options) for instancia in list_path)
    else:
        pool = ProcessPoolExecutor(max_workers=N_WORKERS_INSTANCES, mp_context=mp.get_context("spawn"))
        futures = {pool.submit(process_instance, instancia, centralities, conditions, local_search_options): instancia
                   for instancia in list_path}
        results = (future.result() for future in as_completed(futures))

    for instance_results in results:
        for tag, (df_iteration_rows, summary_rows) in instance_results.items():
            append_csv(df_iteration_rows, filenames[tag], first_write[tag])
            append_csv(summary_rows, summary_filenames[tag], first_write[tag])
            first_write[tag] = False
            total_rows += len(df_iteration_rows)
            for row in summary_rows:
                print(f"[OK] {row['Instance']} [{tag}] run {row['Run']}: banda {row['Initial Bandwidth']} -> {row['Best Bandwidth']} "
                      f"(melhor acao {row['Best Action']}; RCM {row['RCM Bandwidth']}, LB {row['Lower Bound']}) "
                      f"em {row['Global Time (s)']:.1f}s", flush=True)

    if N_WORKERS_INSTANCES > 1:
        pool.shutdown()

    for tag in tags:
        print(f"[INFO] [{tag}] resultado: {filenames[tag]} | resumo: {summary_filenames[tag]}", flush=True)
    print(f"[INFO] Total de registros globais: {total_rows}", flush=True)
    print("[INFO] Execucao finalizada com sucesso.", flush=True)
