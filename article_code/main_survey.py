# lendo todas as instancias de uma classe

import os
import time
import logging
import multiprocessing as mp
import pandas as pd
import numpy as np
import networkx as nx
import torch
from modules.utils.read_filenames import readFilesInDict
from modules.graph.Grafo import GrafoListaAdj, RedutorGrafo
from agent import Agent
from enviroment import Env
from modules.utils import read_Instances
import matplotlib.pyplot as plt
from modules.utils.handle_labels import set_bandwidth_fast
import json

TIMEOUT_SECONDS = 90  # tempo maximo (s) para ler grafo + computar centralidades

if 'C:\\Temp\\C##\\BandWidths' not in os.getcwd() :
    TIMEOUT_SECONDS = np.inf  # desabilitar timeout se não estiver rodando no servidor

MAX_NODES_REDUCED = 2000  # tamanho do subgrafo caso estoure o timeout
REDUCTION_STRATEGY = "bfs"  # 'bfs' ou 'random'


def configure_logger(log_file_path: str, logger_name: str) -> logging.Logger:
    """Configure logger to write both to console and file."""
    logger = logging.getLogger(logger_name)
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


def _worker_load_and_centralities(instancia_path, centralities, out_queue):
    """Worker rodado em processo separado para permitir timeout (Windows-friendly)."""
    try:
        nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance_fast(instancia_path)

        # NetworkX graph
        G = nx.Graph()
        G.add_edges_from(edges)

        # Centralities (cache)
        centralities_maps = {}
        for centrality_key, centrality in centralities.items():
            centralities_maps[centrality_key] = centrality["func"](G, **centrality["args"])

        out_queue.put({
            "ok": True,
            "nnodes": nnodes,
            "edges": edges,
            "centralities_maps": centralities_maps,
        })
    except Exception as e:
        out_queue.put({"ok": False, "error": repr(e)})


def load_graph_and_centralities_with_timeout(instancia_path, centralities, timeout_seconds):
    """Try to load graph + compute centralities within timeout.

    Returns a dict with keys:
      - ok: bool
      - timed_out: bool
      - nnodes, edges, centralities_maps (when ok)
      - error (when not ok)
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_load_and_centralities, args=(instancia_path, centralities, q))
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "timed_out": True, "error": f"timeout after {timeout_seconds}s"}

    if q.empty():
        return {"ok": False, "timed_out": False, "error": "worker returned no data"}

    resp = q.get()
    resp.setdefault("timed_out", False)
    return resp

# Função para plotar uma matriz esparsa
def plot_sparse_matrix(matrix, title, file_name="saida.png"):
    plt.figure(figsize=(8, 8))
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.savefig(file_name)   # salva em arquivo
    plt.close()              # fecha a figura para não abrir

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # diretorio do script
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Inicio da execucao do main_survey.py ({run_timestamp})")

    list_instance, list_band, list_time, global_iteration = [], [], [], []

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

        "Closeness": {
            "func": nx.closeness_centrality,
            "args": {},
            "reverse": True
        },

        "Harmonic Centrality": {
            "func": nx.harmonic_centrality,
            "args": {},
            "reverse": True
        },

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

        # Katz acelerado
        "Katz Centrality": {
            "func": nx.katz_centrality,
            "args": {"alpha": 0.005, "beta": 1.0, "max_iter": 200, "tol": 1e-2},
            "reverse": True
        },

        # PageRank acelerado
        "PageRank": {
            "func": nx.pagerank,
            "args": {"alpha": 0.85, "max_iter": 100},
            "reverse": True
        }
    }

    # ------------------------------------------------------------
    # LOOP PRINCIPAL — com cache automático das centralidades
    # ------------------------------------------------------------

    todos_movimentos = list(range(len(centralities)))
    centralities_list = list(centralities.keys())

    filename = os.path.join(base_dir, "result_output_survey.csv")
    survey_file = os.path.join(base_dir, "data", "survey", )
    
    dir_list = [nome for nome in os.listdir(survey_file) 
                    if os.path.isdir(os.path.join(survey_file, nome))]
    print(f"[INFO] Total de classes no survey: {len(dir_list)}")
    
    for kind in dir_list:
        path = os.path.join(survey_file, kind)
        list_path = readFilesInDict(path, ".mtx")
        print(f"[INFO] Classe {kind} com {len(list_path)} instancias")

        for instancia in list_path:
            results = {}

            instance_path = os.path.basename(instancia).replace(".mtx", "")
            path_name = os.path.join(os.path.dirname(os.path.dirname(instancia)), "plots", f"{instance_path}")
            log_file_path = os.path.join(os.path.dirname(__file__), "logs",f"{instance_path}.log")

            # criar o diretório se não existir
            os.makedirs(path_name, exist_ok=True)
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            instance_timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            logger_name = f"main_survey_experiment_{instance_path}_{instance_timestamp}"
            logger = configure_logger(log_file_path, logger_name)
            torch_save_path = os.path.join(os.path.dirname(os.path.dirname(instancia)), "plots", f"{instance_path}", f'trained_model_{instance_path}.pth')
            json_save_path = os.path.join(os.path.dirname(os.path.dirname(instancia)), "plots", f"{instance_path}", f'analysis_inputs_{instance_path}.json')
            
            # if os.path.exists(torch_save_path):
            #     print(f"Modelo já treinado para a instância {instance_path}, pulando...")
            #     continue

            logger.info("Inicio da instancia: %s", instancia)
            logger.info("Arquivo de log salvo em: %s", log_file_path)

            # 1) Tenta ler grafo + centralidades dentro do timeout
            resp = load_graph_and_centralities_with_timeout(instancia, centralities, TIMEOUT_SECONDS)
            
            if resp.get("ok"):
                nnodes = resp["nnodes"]
                edges = resp["edges"]
                centralities_maps = resp["centralities_maps"]
                nedges = len(edges)
            else:
                # 2) Fallback: reduz o grafo e recomputa centralidades no subgrafo
                if resp.get("timed_out"):
                    logger.warning("[timeout] %s -> usando grafo reduzido (max_nodes=%s)", resp.get("error"), MAX_NODES_REDUCED)
                else:
                    logger.warning("[erro] ao computar centralidades no grafo completo: %s -> usando grafo reduzido", resp.get("error"))

                nnodes_full, nedges_full, edges_full, neighbours, lista_adj, matrix = read_Instances.load_instance_fast(instancia)
                nnodes, edges, mapa, inv = RedutorGrafo.ReduzirArestas(
                    nnodes_full,
                    edges_full,
                    max_nodes=MAX_NODES_REDUCED,
                    estrategia=REDUCTION_STRATEGY,
                    seed=0,
                )
                nedges = len(edges)

                # NetworkX graph reduzido
                G_reduced = nx.Graph()
                G_reduced.add_edges_from(edges)

                centralities_maps = {}
                for centrality_key, centrality in centralities.items():
                    centralities_maps[centrality_key] = centrality["func"](G_reduced, **centrality["args"])
            
            #parametros
            max_iter = 30

            # Instancia grafo NetworkX
            G = nx.Graph()
            G.add_edges_from(edges)

            temp_bandwidth = set_bandwidth_fast(G)
            logger.info("Bandwidth original: %s", temp_bandwidth)

            start_time = time.time()

            # Instancia grafo da sua própria lib
            grafo = GrafoListaAdj()
            grafo.DefinirN(nnodes, VizinhancaDuplamenteLigada=True)
            for (u, v) in edges:
                grafo.AdicionarAresta(u, v)

            #Agent to learning and trainning
            agent = Agent(learning_rate=0.001,
                        gamma=0.9,
                        epsilon=1,
                        eps_min=0.01,
                        eps_dec=0.995,
                        n_movements=len(centralities),
                        n_actions=len(todos_movimentos),
                        n_states=4,
                        deep=True)

            env = Env(100000)

            state = env.get_initial_state()
            agent_movements = agent.get_movements()
            score = 0
            t=0

            local_iteration = []
            start_time = time.time()

            for i in range(max_iter):
                env.reset()
                t = i + 2
                #Escolha da centralidade e da lista
                action = agent.choose_action(state)
                centrality = centralities_list[action]

                centrality_dict_values = centralities_maps[centrality]
                
                info = env.step(graph=grafo, centrality_values=centrality_dict_values, cent_str=centrality, centralities=centralities)
                n_step = info["n_steps"]
                grafo_solution = info["graph"]
                solution = info["solution"]
                reward = info["reward"]
                gap = info["gap"]
                bandwidth = info["bandwidth"]
                local_iteration.extend(info["iterations"])
                
                new_state = [n_step, gap, reward, bandwidth]
                
                #Treinamento do novo estado
                agent.learn(state, action, reward, new_state)

                #score acumulado por iteração
                score += reward

                logger.info(
                    ">> Episode %s Gap: %s Reward: %s Score: %s Centrality: %s Bandwidth: %s",
                    i + 1,
                    gap,
                    reward,
                    score,
                    centrality,
                    bandwidth,
                )
                state = new_state

                if temp_bandwidth > bandwidth:
                    name_matrix = f"bandwidth_{bandwidth}_{centrality}"
                    file_name = os.path.basename(instancia).replace(".mtx", f"_{name_matrix}.png")
                    
                    # salvar a melhor solução
                    temp_bandwidth = bandwidth
                    grafo = grafo_solution

                    G_reordered = nx.relabel_nodes(G, solution)
                    adj_matrix_reordered = nx.to_numpy_array(G_reordered) 
                    plot_sparse_matrix(adj_matrix_reordered, name_matrix, file_name=os.path.join(path_name, file_name))
                    logger.info("Melhor solução atualizada. Plot salvo: %s", os.path.join(path_name, file_name))

            end_time = time.time()
            global_time = end_time - start_time
            df_iteration = pd.DataFrame(local_iteration)
            df_iteration["Instance"] = instance_path
            df_iteration["Edges"] = nedges
            df_iteration["Nodes"] = nnodes
            # df_iteration["NumOfComp"] = num_components
            # df_iteration["LargestCompSize"] = largest_size
            # df_iteration["Diameter"] = diameter
            # df_iteration["Node Connectivity"] = results.get("Node Connectivity", None)
            # df_iteration["Edge Connectivity"] = results.get("Edge Connectivity", None)
            # df_iteration["Algebraic Connectivity"] = results.get("Algebraic Connectivity", None)
            # df_iteration["Average Node Connectivity"] = results.get("Average Node Connectivity", None)
            # df_iteration["Graph Density"] = results.get("Graph Density", None)
            # df_iteration["Average Shortest Path Length"] = results.get("Average Shortest Path Length", None)
            df_iteration['Global Time (s)'] = global_time
            df_iteration_dict = df_iteration.to_dict(orient='records')

            with open(json_save_path, 'w') as f:
                json.dump(df_iteration_dict, f, indent=4)
            logger.info("Analise por instancia salva em: %s", json_save_path)
              
            torch.save(agent.Q.state_dict(), torch_save_path)
            logger.info("Modelo salvo em: %s", torch_save_path)

            global_iteration.extend(df_iteration_dict)
            # todos_movimentos = list(range(len(centralities)))

            # custo_s = centrality_heuristic(graph=grafo_adj, centrality_values=centralities_maps[centralities[0]], cent_str=centralities[0], alpha=0.3, iter_max=50, centralities=centralities)
            # print("Banda Final multicetrality: ", custo_s)
    df_global = pd.DataFrame(global_iteration)
    df_global.to_csv(filename, index=False)
    print(f"[INFO] Resultado global salvo em: {filename}")
    print(f"[INFO] Total de registros globais: {len(df_global)}")
    print("[INFO] Execucao finalizada com sucesso.")

