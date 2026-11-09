# Calcula baselines por instancia para comparar com os resultados do Q-MCH.
#
# Para cada instancia em data/newdata/plots e data/survey/plots o grafo usado na
# execucao e reconstruido exatamente como em main_survey.py (maior componente
# conexa reetiquetada 1..n, ou subgrafo BFS de 2000 nos com seed=0) e sao
# calculados:
#   - banda original (rotulacao do arquivo .mtx)
#   - banda RCM (scipy, estado da pratica segundo o survey)
#   - limites inferiores (Lam et al. 1997 usado no SBPO, ceil(grau_max/2), (n-1)/diametro)
#   - baseline aleatorio: construtivo do VNS-band (BFS com rotacao aleatoria),
#     mesmo orcamento de construcoes da execucao (450)
#
# Le os resultados de main_survey.py em plots/<inst>/analysis_inputs_<inst>_<tag>.json
# (tag padrao: "centrality", o Q-MCH original sem busca local).
# Linhas ja existentes em baselines.csv sao recalculadas quando o grafo da execucao
# mudou (outro numero de nos/arestas ou grafo completo vs subgrafo).
#
# Saida: comparison/baselines.csv (reaproveitado por build_report.py)

import os
import sys
import json
import math
import time
import random
import argparse

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.utils import read_Instances
from modules.graph.Grafo import GrafoListaAdj, RedutorGrafo
from modules.components.constructive import init_Solution
from modules.utils.handle_labels import Bf_graph

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines.csv")
DATASETS = {
    "newdata": os.path.join(BASE_DIR, "data", "newdata"),
    "survey": os.path.join(BASE_DIR, "data", "survey"),
}
MAX_NODES_REDUCED = 2000
RUN_TAG = "centrality"  # sufixo dos arquivos de main_survey.py (condicao, '_ls' com busca local)


def run_file_path(plots_dir, instance, tag=RUN_TAG):
    """Arquivo de resultados por construcao gravado por main_survey.py."""
    return os.path.join(plots_dir, instance, f"analysis_inputs_{instance}_{tag}.json")


def run_graph_info(first_row):
    """(nos, arestas, subgrafo?) do grafo usado na execucao."""
    nodes, edges = int(first_row["Nodes"]), int(first_row["Edges"])
    reduced = first_row.get("Reduced Subgraph")
    if reduced is None:  # arquivos antigos sem a coluna
        reduced = nodes == MAX_NODES_REDUCED
    return nodes, edges, bool(reduced)
DIAMETER_MAX_NODES = 5000


def find_mtx(dataset_dir, instance):
    """Procura <instance>.mtx em qualquer subpasta (exceto plots)."""
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith("plots")]
        if f"{instance}.mtx" in files:
            return os.path.join(root, f"{instance}.mtx")
    return None


def build_run_graph(mtx_path, reduced):
    """Reproduz o grafo efetivamente usado pela execucao."""
    nnodes, _, edges, *_ = read_Instances.load_instance_fast(mtx_path)
    full = nx.Graph()
    full.add_nodes_from(range(1, nnodes + 1))
    full.add_edges_from(edges)
    lcc = max(nx.connected_components(full), key=len)
    full_info = {"full_nodes": nnodes, "full_edges": full.number_of_edges(), "lcc_nodes": len(lcc)}

    if reduced and len(lcc) > MAX_NODES_REDUCED:
        k, red_edges, _, _ = RedutorGrafo.ReduzirArestas(
            nnodes, edges, max_nodes=MAX_NODES_REDUCED, estrategia="bfs", seed=0
        )
        graph = nx.Graph()
        graph.add_nodes_from(range(1, k + 1))
        graph.add_edges_from(red_edges)
        return graph, True, full_info

    mapping = {old: new for new, old in enumerate(sorted(lcc), start=1)}
    graph = nx.relabel_nodes(full.subgraph(lcc).copy(), mapping)
    return graph, False, full_info


def bandwidth_of(edges_arr):
    return int(np.abs(edges_arr[:, 0] - edges_arr[:, 1]).max()) if len(edges_arr) else 0


def rcm_bandwidth(graph):
    nodes = sorted(graph.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    e = np.array([(idx[u], idx[v]) for u, v in graph.edges()], dtype=np.int64)
    n = len(nodes)
    data = np.ones(2 * len(e))
    mat = csr_matrix((data, (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    t0 = time.perf_counter()
    perm = reverse_cuthill_mckee(mat, symmetric_mode=True)
    elapsed = time.perf_counter() - t0
    pos = np.empty(n, dtype=np.int64)
    pos[perm] = np.arange(n)
    return bandwidth_of(pos[e]), elapsed


def lower_bounds(graph):
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    disc = (2 * n - 1) ** 2 - 8 * m
    lam = math.ceil(n - (1 + math.sqrt(disc)) / 2) if disc >= 0 else 0
    deg = math.ceil(max(d for _, d in graph.degree()) / 2)
    diam_lb, diameter = 0, None
    if n <= DIAMETER_MAX_NODES:
        diameter = nx.diameter(graph)
        diam_lb = math.ceil((n - 1) / diameter)
    return {"lb_lam": lam, "lb_degree": deg, "lb_diameter": diam_lb, "diameter": diameter,
            "lower_bound": max(lam, deg, diam_lb), "max_degree": max(d for _, d in graph.degree())}


def random_order_baseline(graph, n_constructions, seed=0):
    random.seed(seed)
    grafo = GrafoListaAdj()
    grafo.DefinirN(graph.number_of_nodes(), VizinhancaDuplamenteLigada=True)
    for u, v in graph.edges():
        grafo.AdicionarAresta(u, v)
    values = []
    t0 = time.perf_counter()
    for _ in range(n_constructions):
        values.append(Bf_graph(graph=grafo, F_labels=init_Solution(grafo)))
    elapsed = time.perf_counter() - t0
    return {"random_best": min(values), "random_mean": float(np.mean(values)),
            "random_std": float(np.std(values)), "random_time_s": elapsed,
            "random_n": n_constructions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-random-nodes", type=int, default=60000,
                        help="nao roda o baseline aleatorio acima deste numero de nos")
    parser.add_argument("--only", nargs="*", help="restringe a estas instancias")
    parser.add_argument("--tag", default=RUN_TAG,
                        help="sufixo dos resultados de main_survey.py (ex.: centrality, centrality_ls, alpha)")
    parser.add_argument("--force", action="store_true", help="recalcula todas as instancias")
    args = parser.parse_args()

    done = pd.read_csv(OUT_FILE) if os.path.exists(OUT_FILE) and not args.force else pd.DataFrame()
    cached = {(r["dataset"], r["instance"]): r for r in done.to_dict("records")}

    for dataset, ddir in DATASETS.items():
        plots = os.path.join(ddir, "plots")
        for instance in sorted(os.listdir(plots)):
            run_file = run_file_path(plots, instance, args.tag)
            if not os.path.exists(run_file):
                continue
            if args.only and instance not in args.only:
                continue
            with open(run_file, encoding="utf-8") as f:
                first = json.load(f)[0]
            run_nodes, run_edges, run_reduced = run_graph_info(first)
            old = cached.get((dataset, instance))
            if old is not None:
                if (int(old["run_nodes"]), int(old["run_edges"]), bool(old["reduced_subgraph"])) ==                         (run_nodes, run_edges, run_reduced):
                    continue  # baseline ja calculado sobre o mesmo grafo
                print(f"[INFO] {dataset}/{instance}: grafo da execucao mudou "
                      f"({int(old['run_nodes'])} -> {run_nodes} nos) - recalculando", flush=True)
            mtx = find_mtx(ddir, instance)
            if mtx is None:
                print(f"[WARN] .mtx nao encontrado: {instance}", flush=True)
                continue
            t0 = time.perf_counter()
            graph, reduced, full_info = build_run_graph(mtx, run_reduced)
            if graph.number_of_nodes() != run_nodes:
                print(f"[WARN] {dataset}/{instance}: grafo reconstruido tem {graph.number_of_nodes()} nos, "
                      f"execucao usou {run_nodes}", flush=True)
            e = np.array(list(graph.edges()), dtype=np.int64)
            row = {
                "dataset": dataset,
                "instance": instance,
                "folder": os.path.basename(os.path.dirname(mtx)),
                "reduced_subgraph": reduced,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "run_nodes": run_nodes,
                "run_edges": run_edges,
                **full_info,
                "original_bw": bandwidth_of(e),
            }
            row["rcm_bw"], row["rcm_time_s"] = rcm_bandwidth(graph)
            row.update(lower_bounds(graph))
            if graph.number_of_nodes() <= args.max_random_nodes:
                row.update(random_order_baseline(graph, n_constructions=450))
            cached[(dataset, instance)] = row  # substitui a linha antiga, se houver
            pd.DataFrame(list(cached.values())).to_csv(OUT_FILE, index=False)
            print(f"[OK] {dataset}/{instance} n={row['nodes']} m={row['edges']} "
                  f"orig={row['original_bw']} rcm={row['rcm_bw']} "
                  f"rand={row.get('random_best')} lb={row['lower_bound']} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
