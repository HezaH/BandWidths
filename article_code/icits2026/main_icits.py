# Reproducao do artigo ICITS 2026:
#   "A Matched-Budget Assessment of DQN Centrality Selection for Matrix Bandwidth Minimization"
#   (manuscrito em article_code/2026-ICITS-Peru-3/samplepaper.tex)
#
# Baseado no main_survey.py (mesma organizacao, nomes e funcoes auxiliares), com a configuracao
# descrita no artigo:
#   - 7 centralidades (Degree, Closeness, Betweenness, Eigenvector, Katz, PageRank, Harmonic);
#   - construtivo com LCR sobre o nivel inteiro (construction = "paper"), alpha = 0.3, 15 construcoes
#     por decisao, vertice inicial proporcional a centralidade (uniforme se os pesos forem invalidos);
#   - DQN com estado de 12 dimensoes, recompensa densa, replay e rede alvo (agent_dqn.py);
#   - 300 decisoes de treino + 60 gulosas = 360 x 15 = 5.400 construcoes por execucao;
#   - mesmo orcamento e as mesmas 30 sementes para DQN, selecao aleatoria e cada centralidade fixa;
#   - ablacao 2x2: seletor (DQN, aleatorio) x vertice inicial (centralidade, pseudo-periferico).
#
# Extensoes propostas na conclusao do artigo (opcoes novas; os padroes reproduzem o artigo):
#   E1 --construction parent      -> filhos de cada vertice rotulados na ordem do pai (Cuthill-McKee)
#   E2 --reward half_unit_tie     -> empate vale 0.5/B0, abaixo de qualquer melhoria estrita
#   E3 --selectors UCB1 AP        -> seletores de bandit (UCB1 e adaptive pursuit), mesma recompensa
#
# Base de dados: --dataset sbpo (13 instancias HB do artigo, data/newdata) ou survey (data/survey).
# Sem --dataset, a escolha e perguntada no terminal (como no main_survey.py).
#
# Saidas: instances.csv, runs.csv (uma linha por execucao) e decisions.csv (uma linha por decisao),
# em icits2026/results/ (sbpo + construcao do artigo) ou icits2026/results_<dataset>_<construcao>/.
# O relatorio e gerado por report_icits.py.
#
# Paralelismo: sempre em paralelo. Por padrao (--workers auto) usa os nucleos livres no momento (nucleos
# logicos x (1 - uso medio da CPU)) menos 2 (--reserve-cores), tanto na preparacao das instancias quanto nas
# execucoes. Para rodar varias etapas em sequencia, veja pipeline_icits.py.
#
# Exemplos:
#   python icits2026/main_icits.py                                  -> pergunta a base; principal + ablacao
#   python icits2026/main_icits.py --dataset sbpo       -> reproducao do artigo
#   python icits2026/main_icits.py --dataset sbpo --construction parent           -> E1
#   python icits2026/main_icits.py --dataset sbpo --experiment main --selectors DQN \
#          --reward half_unit_tie                                                -> E2
#   python icits2026/main_icits.py --dataset sbpo --experiment main --selectors UCB1 AP  -> E3
#   python icits2026/main_icits.py --dataset sbpo --train 20 --greedy 4 --seeds 2 --results-dir icits2026/results_teste
#                                                                   -> teste rapido (orcamento reduzido)

if __name__ == "__main__":  # processos filhos (spawn) reimportam este arquivo
    print("[INFO] Iniciando imports do main_icits.py", flush=True)

import os
import sys
import time
import math
import pickle
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import torch
from networkx.utils import reverse_cuthill_mckee_ordering
from scipy.sparse.csgraph import shortest_path

HERE = os.path.dirname(os.path.abspath(__file__))   # article_code/icits2026
BASE_DIR = os.path.dirname(HERE)                     # article_code
for _p in (BASE_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modules.graph.Grafo import GrafoListaAdj
from modules.utils import read_Instances
from modules.utils.handle_labels import set_bandwidth_fast
from main_survey import (_connected_graph_with_mapping, configure_logger, bandwidth_of_solution,
                         lower_bound_bandwidth, seed_everything)
from centralities import closeness_centrality_aprox
from agent_dqn import DQNAgent
from environment_icits import EnvICITS, REWARD_SCHEMES
from bandits import UCB1Selector, AdaptivePursuitSelector

# ------------------------------------------------------------
# PARAMETROS DO EXPERIMENTO (Secao 3.2 e 4.1 do artigo)
# ------------------------------------------------------------
TRAIN_DECISIONS = 300   # T: decisoes de treino (epsilon do DQN decai ate eps_min em T; bandits aprendem ate T)
GREEDY_DECISIONS = 60   # decisoes gulosas (sem exploracao e sem aprendizado) apos o treino
N_CALLS = 5             # chamadas de centrality_heuristic por decisao (Env.step)
ITER_MAX = 3            # construcoes por chamada -> N_CALLS x ITER_MAX = 15 construcoes por decisao
ALPHA = 0.3             # tamanho relativo da lista restrita de candidatos (LCR)
N_SEEDS = 30            # sementes deterministicas por (instancia, metodo)
BASE_SEED = 0           # semente da execucao r = BASE_SEED + r (mesmas sementes para todos os metodos)

# ------------------------------------------------------------
# BASES DE DADOS
# ------------------------------------------------------------
DATASETS = {"sbpo": "newdata", "artigo": "newdata", "newdata": "newdata", "survey": "survey"}  # opcao -> pasta
DATASET_KEY = {"sbpo": "sbpo", "artigo": "sbpo", "newdata": "sbpo", "survey": "survey"}       # opcao -> chave
# 13 instancias do grupo HB do SuiteSparse (Secao 4.1). bcsstk02 (K66) e bcsstk07 (identica a
# bcsstk06) ficam de fora. Os nomes seguem os arquivos de data/newdata (lshp_265 = lshp265 no artigo).
INSTANCES = ["bcsstk01", "bcsstk04", "bcsstk05", "bcsstk06",
             "lshp_265", "lshp_406", "lshp1009", "lshp1270", "lshp1561", "lshp1882",
             "bp_0", "bp_200", "shl_0"]
# Instancias discriminantes usadas na ablacao (Secao 4.3): bcsstk04/05/06 dao B0 em todas as sementes.
ABLATION_INSTANCES = ["bcsstk01", "bp_0", "bp_200", "shl_0",
                      "lshp_265", "lshp_406", "lshp1009", "lshp1270", "lshp1561", "lshp1882"]

# ------------------------------------------------------------
# CONSTRUTIVO (extensao E1)
# ------------------------------------------------------------
# "paper":  cada nivel inteiro e ordenado pela centralidade e embaralhado pela LCR (Secao 3.1 do artigo)
# "parent": os filhos de cada vertice sao rotulados na ordem do pai (regra de Cuthill-McKee) e a LCR
#           ordena so os filhos de cada pai -- "parent-preserving within-level ordering" (conclusao)
CONSTRUCTIONS = {"paper": False, "parent": True}  # construcao -> parent_order do construtivo

# ------------------------------------------------------------
# CENTRALIDADES (acoes do agente)
# ------------------------------------------------------------
# O artigo nao fixa parametros; as configuracoes abaixo vem do main_survey.py (inclusive as que
# la estao comentadas). Na base SBPO, Closeness, Harmonic e Betweenness sao exatas: os grafos tem no
# maximo 1.882 vertices e as centralidades sao calculadas uma unica vez por instancia (cache).


def harmonic_centrality_aprox(graph, k=512, seed=0, chunk=32):
    """Harmonic aproximada por amostragem de k vértices-fonte, no mesmo esquema da
    closeness_centrality_aprox de centralities.py (BFS em C pelo scipy). Com k >= n é exata.
    A escala difere da do networkx por um fator constante, o que não muda ordenação nem sorteio."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return {v: 0.0 for v in nodes}
    matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, format="csr")
    rng = np.random.default_rng(seed)
    sources = np.arange(n) if k is None or k >= n else rng.choice(n, size=k, replace=False)
    inv_sum = np.zeros(n)
    for start in range(0, len(sources), chunk):
        dist = shortest_path(matrix, directed=False, unweighted=True, indices=sources[start:start + chunk])
        with np.errstate(divide="ignore"):
            inv = np.where((dist > 0) & np.isfinite(dist), 1.0 / dist, 0.0)
        inv_sum += inv.sum(axis=0)
    return dict(zip(nodes, inv_sum.tolist()))


CENTRALITIES = {
    "Degree": {"func": nx.degree_centrality, "args": {}, "reverse": True},
    "Closeness": {"func": nx.closeness_centrality, "args": {}, "reverse": True},
    "Betweenness": {"func": nx.betweenness_centrality, "args": {}, "reverse": True},
    "Eigenvector": {"func": nx.eigenvector_centrality, "args": {"max_iter": 1000}, "reverse": True},
    "Katz": {"func": nx.katz_centrality, "args": {"alpha": 0.005, "beta": 1.0, "max_iter": 1000},
             "reverse": True},
    "PageRank": {"func": nx.pagerank, "args": {"alpha": 0.85, "max_iter": 100}, "reverse": True},
    "Harmonic": {"func": nx.harmonic_centrality, "args": {}, "reverse": True},
}
# Base survey (grafos de ate ~534 mil vertices): mesmas 7 acoes, com as versoes aproximadas do
# main_survey.py para as centralidades baseadas em caminhos (exatas custariam dias).
CENTRALITIES_SURVEY = {
    **CENTRALITIES,
    "Closeness": {"func": closeness_centrality_aprox, "args": {"k": 512}, "reverse": True},
    "Betweenness": {"func": nx.betweenness_centrality, "args": {"k": 50, "seed": 0}, "reverse": True},
    "Harmonic": {"func": harmonic_centrality_aprox, "args": {"k": 512}, "reverse": True},
}
CENTRALITY_NAMES = list(CENTRALITIES)


def centrality_config(dataset):
    return CENTRALITIES_SURVEY if dataset == "survey" else CENTRALITIES


# ------------------------------------------------------------
# METODOS (seletor [+ recompensa] + regra do vertice inicial)
# ------------------------------------------------------------
# "DQN" / "Random" / "Fixed:<centralidade>" / "UCB1" / "AP" usam o inicio proporcional a centralidade;
# o sufixo "-PP" troca o inicio por um vertice pseudo-periferico (ablacao da Secao 4.3) e o sufixo
# "@<esquema>" indica uma recompensa diferente da do artigo (so nos seletores que aprendem).
# Ex.: "DQN", "DQN-PP", "DQN@half_unit_tie", "UCB1@half_unit_tie-PP".
LEARNING_SELECTORS = ("DQN", "UCB1", "AP")
SELECTOR_CHOICES = ("DQN", "Random", "Fixed", "UCB1", "AP")
DEFAULT_MAIN_SELECTORS = ["DQN", "Random", "Fixed"]  # experimento principal do artigo
DEFAULT_ABLATION_SELECTORS = ["DQN", "Random"]       # ablacao do artigo
MAIN_METHODS = ["DQN", "Random"] + [f"Fixed:{c}" for c in CENTRALITY_NAMES]
ABLATION_METHODS = ["DQN-PP", "Random-PP"]


def method_name(selector, reward="paper", pp=False):
    """Monta o nome do método; o esquema de recompensa só aparece quando difere do artigo."""
    name = f"{selector}@{reward}" if selector in LEARNING_SELECTORS and reward != "paper" else selector
    return name + ("-PP" if pp else "")


def build_methods(selectors, rewards, pp):
    methods = []
    for s in selectors:
        if s == "Fixed":
            methods += [method_name(f"Fixed:{c}", pp=pp) for c in CENTRALITY_NAMES]
        elif s in LEARNING_SELECTORS:
            methods += [method_name(s, r, pp) for r in rewards]
        else:
            methods.append(method_name(s, pp=pp))
    return methods


def parse_method(method):
    """'DQN@half_unit_tie-PP' -> ('DQN', 'half_unit_tie', 'peripheral');
    'Fixed:Katz' -> ('Fixed:Katz', 'paper', 'centrality')."""
    start = "centrality"
    if method.endswith("-PP"):
        method, start = method[:-3], "peripheral"
    selector, _, reward = method.partition("@")
    return selector, reward or "paper", start


# ------------------------------------------------------------
# PARALELISMO E SAIDAS
# ------------------------------------------------------------
RESERVED_CORES = 2                              # nucleos deixados livres: processos = nucleos livres - 2
CACHE_DIR = os.path.join(HERE, "cache")         # grafo + centralidades por instancia
RESULTS_DIR = os.path.join(HERE, "results")     # sbpo + construcao do artigo (reproducao)
LOGS_DIR = os.path.join(HERE, "logs")


def default_results_dir(dataset, construction):
    """results/ guarda a reprodução do artigo; as demais combinações ficam em pastas próprias."""
    if dataset == "sbpo" and construction == "paper":
        return RESULTS_DIR
    return os.path.join(HERE, f"results_{dataset}_{construction}")


def cache_dir(dataset):
    return CACHE_DIR if dataset == "sbpo" else os.path.join(CACHE_DIR, dataset)


def ask_dataset():
    """Pergunta no terminal qual base rodar, como no main_survey.py; devolve a chave ('sbpo' ou 'survey')."""
    options = {"1": "sbpo", "2": "survey"}
    print("\nQual base de dados deseja rodar?")
    print("  [1] SBPO / artigo  (13 instancias HB do artigo ICITS, data/newdata)")
    print("  [2] Survey         (matrizes de data/survey; centralidades aproximadas e custo muito alto)")
    while True:
        answer = input("Escolha 1 ou 2: ").strip().lower()
        if answer in options:
            return options[answer]
        if answer in DATASET_KEY:
            return DATASET_KEY[answer]
        print("Opcao invalida. Digite 1 (SBPO) ou 2 (survey).")


def find_instance_file(name, dataset="sbpo"):
    """Procura <name>.mtx na pasta da base (fora de plots/ e trash/, que contém cópias)."""
    root_dir = os.path.join(BASE_DIR, "data", DATASETS[dataset])
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("plots") and d != "trash")
        if f"{name}.mtx" in files:
            return os.path.join(root, f"{name}.mtx")
    raise FileNotFoundError(f"{name}.mtx não encontrado em {root_dir}")


def dataset_instances(dataset, experiment):
    """Instâncias de cada experimento. Na base survey, todas as matrizes de data/survey (a ablação
    usa as mesmas, porque não se sabe de antemão quais são discriminantes)."""
    if dataset == "sbpo":
        return INSTANCES if experiment == "main" else ABLATION_INSTANCES
    names = []
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "data", DATASETS[dataset])):
        dirs[:] = sorted(d for d in dirs if not d.startswith("plots"))
        names += sorted(f[:-4] for f in files if f.endswith(".mtx"))
    return list(dict.fromkeys(names))


# ------------------------------------------------------------
# METRICAS DE REFERENCIA
# ------------------------------------------------------------

def rcm_bandwidth_networkx(G):
    """Banda do Reverse Cuthill-McKee do networkx. Reproduz os valores de RCM da Tabela 3 do artigo
    (o RCM do scipy usado no main_survey.py diverge, ex.: lshp1270 65 x 38)."""
    position = {v: i for i, v in enumerate(reverse_cuthill_mckee_ordering(G))}
    return max((abs(position[u] - position[v]) for u, v in G.edges()), default=0)


def compute_centrality(G, name, dataset="sbpo"):
    """Calcula uma centralidade; Eigenvector e Katz recorrem à versão numpy se não convergirem."""
    centrality = centrality_config(dataset)[name]
    try:
        return centrality["func"](G, **centrality["args"])
    except nx.PowerIterationFailedConvergence:
        fallback = {"Eigenvector": lambda g: nx.eigenvector_centrality_numpy(g),
                    "Katz": lambda g: nx.katz_centrality_numpy(g, alpha=centrality["args"]["alpha"],
                                                                beta=centrality["args"]["beta"])}
        return fallback[name](G)


def valid_start_weights(values):
    """O artigo sorteia o vértice inicial proporcionalmente à centralidade e volta ao sorteio
    uniforme quando os pesos são não finitos, negativos ou têm soma zero."""
    w = np.asarray(list(values), dtype=float)
    return bool(np.all(np.isfinite(w)) and np.all(w >= 0) and w.sum() > 0)


# ------------------------------------------------------------
# ETAPAS DO FLUXO POR INSTANCIA
# ------------------------------------------------------------

def prepare_instance(name, dataset="sbpo", refresh=False):
    """Lê a instância (maior componente conexa, rótulos 1..n), calcula as 7 centralidades, B0, RCM
    e limite inferior, e guarda tudo em cache/[<base>/]<name>.pkl (reaproveitado por todas as execuções)."""
    folder = cache_dir(dataset)
    os.makedirs(folder, exist_ok=True)
    cache_path = os.path.join(folder, f"{name}.pkl")
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    path = find_instance_file(name, dataset)
    nnodes, nedges, edges, *_ = read_Instances.load_instance_fast(path)
    G, node_mapping, inverse_mapping = _connected_graph_with_mapping(edges, nnodes)
    G.remove_edges_from(nx.selfloop_edges(G))

    centralities_maps, centrality_times = {}, {}
    for c in CENTRALITY_NAMES:
        start = time.time()
        centralities_maps[c] = compute_centrality(G, c, dataset)
        centrality_times[c] = time.time() - start

    data = {
        "Instance": name,
        "dataset": dataset,
        "path": path,
        "nnodes": G.number_of_nodes(),
        "edges": list(G.edges()),
        "centralities_maps": centralities_maps,
        "centrality_times": centrality_times,
        "valid_start": {c: valid_start_weights(m.values()) for c, m in centralities_maps.items()},
        "inverse_mapping": inverse_mapping,
        "info": {
            "Instance": name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Initial Bandwidth": set_bandwidth_fast(G),          # B0
            "RCM Bandwidth": rcm_bandwidth_networkx(G),
            "Lower Bound": lower_bound_bandwidth(G),
            "Centrality Time (s)": sum(centrality_times.values()),
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data


_PROCESS_CACHE = {}  # (base, instância) -> (data, grafo, G); um por processo


def get_instance(name, dataset="sbpo"):
    key = (dataset, name)
    if key not in _PROCESS_CACHE:
        data = prepare_instance(name, dataset)
        grafo = GrafoListaAdj()
        grafo.DefinirN(data["nnodes"], VizinhancaDuplamenteLigada=True)
        for (u, v) in data["edges"]:
            grafo.AdicionarAresta(u, v)
        G = nx.Graph()
        G.add_edges_from(data["edges"])
        _PROCESS_CACHE[key] = (data, grafo, G)
    return _PROCESS_CACHE[key]


def reset_random_caches(grafo):
    """O conjunto de vértices pseudo-periféricos é sorteado e guardado no grafo na 1ª construção.
    Ele é descartado antes de cada execução para depender só da semente da execução (e não de qual
    execução usou o grafo primeiro no mesmo processo)."""
    if hasattr(grafo, "_peripheral_pool_cache"):
        del grafo._peripheral_pool_cache


# ------------------------------------------------------------
# UMA EXECUCAO (seletor x semente)
# ------------------------------------------------------------

def run_selector(grafo, data, method, seed, train_decisions, greedy_decisions, save_decisions=True,
                 construction="paper", dataset="sbpo"):
    """Uma execução de `method` com `seed`: (train + greedy) decisões de 15 construções.

    Todos os métodos recebem o mesmo orçamento e começam com B0 como incumbente. Os seletores que
    aprendem (DQN, UCB1, AP) aprendem só nas `train_decisions` primeiras decisões; nas `greedy_decisions`
    seguintes escolhem a ação de maior estimativa, sem explorar. Random e Fixed usam as mesmas decisões
    sem distinção de fase.

    Retorna (linha de resumo, linhas por decisão).
    """
    seed_everything(seed)
    reset_random_caches(grafo)
    selector, reward_scheme, start_rule = parse_method(method)
    actions = CENTRALITY_NAMES
    total = train_decisions + greedy_decisions
    env = EnvICITS(data["info"]["Initial Bandwidth"], n_actions=len(actions), horizon=train_decisions,
                   alpha=ALPHA, n_calls=N_CALLS, iter_max=ITER_MAX, parent_order=CONSTRUCTIONS[construction],
                   reward_scheme=reward_scheme)
    agent = DQNAgent(n_states=5 + len(actions), n_actions=len(actions), horizon=train_decisions,
                     seed=seed) if selector == "DQN" else None
    # geradores próprios para a seleção aleatória e os bandits: não consomem o random/np.random do construtivo
    policy_rng = random.Random(seed + 1_000_003)
    bandit_rng = np.random.default_rng(seed + 2_000_003)
    bandit = {"UCB1": UCB1Selector, "AP": AdaptivePursuitSelector}.get(selector)
    bandit = bandit(len(actions), bandit_rng) if bandit else None
    fixed_action = actions.index(selector.split(":", 1)[1]) if selector.startswith("Fixed:") else None
    if agent is None and bandit is None and fixed_action is None and selector != "Random":
        raise ValueError(f"Método inválido: {method}")

    state = env.get_state()
    best_train = env.best_sol
    greedy_min = math.inf
    decision_rows = []
    start_time = time.time()
    for t in range(total):
        greedy_phase = t >= train_decisions
        if agent is not None:
            action = agent.choose_action(state, t, greedy=greedy_phase)
        elif bandit is not None:
            action = bandit.choose(greedy=greedy_phase)
        elif fixed_action is not None:
            action = fixed_action
        else:
            action = policy_rng.randrange(len(actions))
        centrality = actions[action]
        start_strategy = start_rule if start_rule == "peripheral" else (
            "centrality" if data["valid_start"][centrality] else "uniform")

        info = env.step(grafo, action, data["centralities_maps"][centrality], centrality, CENTRALITIES,
                        start_strategy=start_strategy)
        next_state = env.get_state()

        #Treinamento (só os seletores que aprendem, só na fase de treino); a última transição é terminal
        if not greedy_phase:
            if agent is not None:
                agent.remember(state, action, info["reward"], next_state, done=(t == train_decisions - 1))
                agent.learn()
            elif bandit is not None:
                bandit.update(action, info["reward"])
        state = next_state

        if greedy_phase:
            greedy_min = min(greedy_min, info["bandwidth"])
        else:
            best_train = env.best_sol
        if save_decisions:
            decision_rows.append({
                "Instance": data["Instance"], "Method": method, "Seed": seed, "Decision": t + 1,
                "Phase": "greedy" if greedy_phase else "train", "Action": centrality,
                "Bandwidth": info["bandwidth"], "Incumbent": info["incumbent"],
                "Reward": round(info["reward"], 6),
                "Epsilon": round(agent.epsilon, 5) if agent is not None else np.nan,
            })
    global_time = time.time() - start_time

    # confere que a rotulação do incumbente reproduz a banda reportada
    if env.best_solution is not None:
        _, _, G = get_instance(data["Instance"], dataset)
        check = bandwidth_of_solution(G, env.best_solution)
        if check != env.best_sol:
            print(f"[WARN] {data['Instance']} {method} seed {seed}: banda da solução ({check}) difere "
                  f"da reportada ({env.best_sol})", flush=True)

    summary = {
        **data["info"],
        "Dataset": dataset,
        "Construction": construction,
        "Method": method,
        "Selector": selector,
        "Reward": reward_scheme if selector in LEARNING_SELECTORS else "",
        "Start": start_rule,
        "Seed": seed,
        "Train Decisions": train_decisions,
        "Greedy Decisions": greedy_decisions,
        "Constructions": total * N_CALLS * ITER_MAX,
        "Best Bandwidth": env.best_sol,
        "Best Train Bandwidth": best_train,
        "Best Greedy-Phase Bandwidth": greedy_min if greedy_decisions else np.nan,
        "Greedy Improved": bool(greedy_decisions and greedy_min < best_train),
        "Normalized Bandwidth": env.best_sol / data["info"]["Initial Bandwidth"],
        "Gradient Updates": agent.n_updates if agent is not None else 0,
        "Target Syncs": agent.n_syncs if agent is not None else 0,
        "Global Time (s)": global_time,
    }
    return summary, decision_rows


def run_job(name, method, seed, train_decisions, greedy_decisions, save_decisions, dataset="sbpo",
            construction="paper"):
    """Tarefa de topo (picklable) para o ProcessPoolExecutor."""
    data, grafo, _ = get_instance(name, dataset)
    return run_selector(grafo, data, method, seed, train_decisions, greedy_decisions, save_decisions,
                        construction=construction, dataset=dataset)


def _watch_parent(parent_pid, parent_created, interval=2.0):
    """Encerra o worker se o processo principal deixar de existir. Sem isso, quando o principal morre por
    outro caminho que não o Ctrl+C (terminal fechado, Gerenciador de Tarefas), os workers ficam órfãos
    ocupando CPU e memória."""
    import psutil
    while True:
        time.sleep(interval)
        try:
            alive = psutil.Process(parent_pid).create_time() == parent_created  # PID pode ser reaproveitado
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            alive = False
        if not alive:
            if os.name == "nt":
                # TerminateProcess: no Windows, os._exit roda as rotinas de saída das DLLs, e as do torch
                # (pool de threads OpenMP) podem travar o processo no meio do encerramento
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
            os._exit(1)


def _worker_init(parent_pid=None, parent_created=None):
    """Cada processo usa 1 thread do torch, aquece a rede (a 1ª chamada é lenta e inflaria o tempo
    da 1ª execução do DQN) e vigia o processo principal (encerra-se se ele morrer)."""
    if parent_pid is not None:
        import threading
        threading.Thread(target=_watch_parent, args=(parent_pid, parent_created), daemon=True).start()
    torch.set_num_threads(1)
    warm = DQNAgent(n_states=5 + len(CENTRALITY_NAMES), n_actions=len(CENTRALITY_NAMES), batch_size=1,
                    learn_start=1)
    warm.remember([0.0] * warm.n_states, 0, 0.0, [0.0] * warm.n_states, False)
    warm.learn()


def repair_csv_tail(filename):
    """Se uma interrupção cortou a última linha no meio da gravação, remove essa linha incompleta
    (o arquivo passa a terminar na última linha completa; a execução correspondente é refeita)."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return
    newline = b"\n"
    with open(filename, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == newline:
            return
        size = f.seek(0, os.SEEK_END)
        pos = max(0, size - (1 << 20))                # procura a última quebra de linha no último 1 MB
        f.seek(pos)
        tail = f.read()
        cut = tail.rfind(newline)
        f.truncate(pos + cut + 1 if cut >= 0 else 0)


WRITE_ATTEMPTS = 12  # tentativas de gravação; esperas de 1, 2, ..., 11 s (~66 s no total)


def with_retry(action, filename, attempts=WRITE_ATTEMPTS, undo_size=None):
    """Executa `action` repetindo em caso de OSError. Em pastas sincronizadas (OneDrive), o arquivo fica
    bloqueado por instantes enquanto é enviado, e a gravação falha com 'Invalid argument' ou 'Permission
    denied'. Se `undo_size` for informado, uma gravação parcial é desfeita (o arquivo volta a esse tamanho)
    antes da nova tentativa, para não duplicar linhas."""
    for k in range(1, attempts + 1):
        try:
            return action()
        except OSError as err:
            if k == attempts:
                raise OSError(f"Não foi possível gravar {filename} após {attempts} tentativas ({err}). Se a pasta "
                              "estiver no OneDrive, pause a sincronização ou use --results-dir fora dele; os "
                              "resultados já gravados são retomados ao rodar o mesmo comando.") from err
            if undo_size is not None:
                try:
                    if os.path.exists(filename) and os.path.getsize(filename) > undo_size:
                        with open(filename, "rb+") as f:
                            f.truncate(undo_size)
                except OSError:
                    pass
            print(f"[AVISO] Falha ao gravar {os.path.basename(filename)} ({err}); nova tentativa em {k} s "
                  f"({k}/{attempts - 1})", flush=True)
            time.sleep(k)


def append_csv(rows, filename):
    """Grava as linhas assim que a execução termina (permite retomar uma execução interrompida).
    As colunas seguem o cabeçalho do arquivo; se surgirem colunas novas (ex.: Dataset/Construction/Reward
    num runs.csv gravado antes das extensões), o arquivo é regravado uma vez com a união das colunas,
    num arquivo temporário que só então substitui o original (uma interrupção não trunca o arquivo).
    Toda operação em disco é repetida se o arquivo estiver bloqueado momentaneamente (with_retry)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    with_retry(lambda: repair_csv_tail(filename), filename)
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with_retry(lambda: df.to_csv(filename, index=False), filename, undo_size=0)
        return
    header = with_retry(lambda: pd.read_csv(filename, nrows=0).columns.tolist(), filename)
    new_cols = [c for c in df.columns if c not in header]
    if new_cols:
        tmp = filename + ".tmp"

        def migrate():
            pd.read_csv(filename).reindex(columns=header + new_cols).to_csv(tmp, index=False)
            os.replace(tmp, filename)
        with_retry(migrate, filename)
        header = header + new_cols
    # o bloco inteiro é gravado de uma vez; se a gravação falhar no meio, o arquivo volta ao tamanho anterior
    text = df.reindex(columns=header).to_csv(header=False, index=False)
    size_before = os.path.getsize(filename)

    def write():
        with open(filename, "a", encoding="utf-8", newline="") as f:
            f.write(text)
    with_retry(write, filename, undo_size=size_before)


def done_jobs(runs_file, train_decisions, greedy_decisions, dataset="sbpo", construction="paper"):
    """Execuções já presentes em runs.csv com o mesmo orçamento, base e construção (para retomar sem
    refazer). Linhas antigas, sem as colunas Dataset/Construction, valem como sbpo + construção do artigo."""
    if not os.path.exists(runs_file):
        return set()
    repair_csv_tail(runs_file)
    runs = pd.read_csv(runs_file)
    runs = runs[(runs["Train Decisions"] == train_decisions) & (runs["Greedy Decisions"] == greedy_decisions)]
    if "Construction" in runs.columns:
        runs = runs[runs["Construction"].fillna("paper") == construction]
    elif construction != "paper":
        return set()
    if "Dataset" in runs.columns:
        runs = runs[runs["Dataset"].fillna("sbpo") == dataset]
    elif dataset != "sbpo":
        return set()
    return set(zip(runs["Instance"], runs["Method"], runs["Seed"]))


# ------------------------------------------------------------
# PARALELISMO: processos = nucleos livres - reserva
# ------------------------------------------------------------

def auto_workers(reserve=RESERVED_CORES, sample_seconds=3.0):
    """Núcleos lógicos livres agora (núcleos x (1 - uso médio da CPU, medido por `sample_seconds`))
    menos `reserve`, no mínimo 1. Sem psutil, considera todos os núcleos livres.
    Retorna (processos, núcleos lógicos, núcleos livres, uso %)."""
    total = os.cpu_count() or 1
    try:
        import psutil
        usage = psutil.cpu_percent(interval=sample_seconds)
    except ImportError:
        usage = 0.0
    free = int(total * (1.0 - usage / 100.0))
    return max(1, free - reserve), total, free, usage


def resolve_workers(value, reserve, n_tasks):
    """'auto' -> núcleos livres - reserva; um número fixa a quantidade. Nunca mais processos que tarefas."""
    if str(value).lower() == "auto":
        workers, total, free, usage = auto_workers(reserve)
        how = (f"{total} nucleos logicos, uso atual {usage:.0f}% (~{free} livres), reserva de {reserve} "
               f"-> {workers} processo(s)")
    else:
        workers = max(1, int(value))
        how = f"{workers} processo(s) definidos em --workers"
    workers = min(workers, max(1, n_tasks))
    return workers, how


def _prepare_job(name, dataset, refresh):
    """Tarefa de topo (picklable): prepara a instância, grava o cache e devolve só o resumo
    (os mapas de centralidade ficam no cache em disco, não voltam pelo pipe)."""
    start = time.time()
    data = prepare_instance(name, dataset, refresh=refresh)
    return {"info": data["info"], "times": data["centrality_times"],
            "invalid": [c for c, ok in data["valid_start"].items() if not ok], "elapsed": time.time() - start}


# ------------------------------------------------------------
# PLANO DE EXECUCOES
# ------------------------------------------------------------

def build_plan(dataset, experiment="all", selectors=None, rewards=("paper",), instances=None, methods=None):
    """Lista de (instância, método) de um experimento, na ordem em que aparecem."""
    experiments = ["main", "ablation"] if experiment == "all" else [experiment]
    plan = []
    for exp in experiments:
        insts = dataset_instances(dataset, exp)
        if exp == "main":
            sel = selectors or DEFAULT_MAIN_SELECTORS
        else:  # a ablação do artigo não inclui centralidades fixas
            sel = [s for s in (selectors or DEFAULT_ABLATION_SELECTORS) if s != "Fixed"]
        meths = build_methods(sel, list(rewards), pp=(exp == "ablation"))
        if instances:
            insts = [i for i in insts if i in instances]
        if methods:
            meths = [m for m in meths if m in methods]
        plan += [(i, m) for i in insts for m in meths]
    return list(dict.fromkeys(plan))


def pending_jobs(plan, n_seeds, results_dir, train, greedy, dataset, construction):
    """Execuções (instância, método, semente) do plano que ainda não estão em runs.csv."""
    seeds = [BASE_SEED + r for r in range(n_seeds)]
    already = done_jobs(os.path.join(results_dir, "runs.csv"), train, greedy, dataset, construction)
    return [(i, m, s) for i, m in plan for s in seeds if (i, m, s) not in already]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Reprodução e extensões do artigo ICITS 2026 (seletores x "
                                                 "seleção aleatória x centralidades fixas, mesmo orçamento)")
    parser.add_argument("--dataset", choices=list(DATASET_KEY), default=None,
                        help="sbpo/artigo = 13 instâncias HB do artigo; survey = matrizes de data/survey. "
                             "Se omitido, a escolha é perguntada no terminal")
    parser.add_argument("--experiment", choices=["main", "ablation", "all"], default="all",
                        help="main = seletores com início pela centralidade; ablation = mesmos seletores com "
                             "início pseudo-periférico (-PP); all = os dois")
    parser.add_argument("--construction", choices=list(CONSTRUCTIONS), default="paper",
                        help="paper = construtivo do artigo; parent = ordem do pai (extensão E1)")
    parser.add_argument("--selectors", nargs="+", choices=SELECTOR_CHOICES, default=None,
                        help="seletores a rodar (padrão: os do artigo -> main: DQN Random Fixed; ablation: DQN "
                             "Random). UCB1 e AP = extensão E3")
    parser.add_argument("--reward", nargs="+", choices=list(REWARD_SCHEMES), default=["paper"],
                        help="esquemas de recompensa dos seletores que aprendem (DQN, UCB1, AP); "
                             "half_unit_tie = extensão E2")
    parser.add_argument("--instances", nargs="*", default=None, help="restringe às instâncias com estes nomes")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="restringe aos métodos (nomes completos, ex.: DQN Random DQN@half_unit_tie)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="sementes por (instância, método)")
    parser.add_argument("--train", type=int, default=TRAIN_DECISIONS, help="decisões de treino (T)")
    parser.add_argument("--greedy", type=int, default=GREEDY_DECISIONS, help="decisões gulosas após o treino")
    parser.add_argument("--workers", default="auto",
                        help="processos em paralelo: 'auto' (padrão) = núcleos livres - reserva; ou um número")
    parser.add_argument("--reserve-cores", type=int, default=RESERVED_CORES,
                        help="núcleos deixados livres no modo auto (padrão: %(default)s)")
    parser.add_argument("--results-dir", default=None,
                        help="pasta de saída (padrão: results/ para sbpo + paper; results_<base>_<construção>/)")
    parser.add_argument("--no-decisions", action="store_true", help="não grava decisions.csv (economiza disco)")
    parser.add_argument("--fresh", action="store_true", help="apaga os resultados anteriores da pasta de saída")
    parser.add_argument("--refresh-cache", action="store_true", help="recalcula grafo e centralidades")
    return parser.parse_args(argv)


# ------------------------------------------------------------
# FLUXO COMPLETO: preparação das instâncias + execuções, tudo no mesmo pool de processos
# ------------------------------------------------------------

def run_experiment(args):
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Inicio da execucao do main_icits.py ({run_timestamp})", flush=True)
    dataset = DATASET_KEY[args.dataset] if args.dataset else ask_dataset()
    results_dir = args.results_dir or default_results_dir(dataset, args.construction)
    print(f"[INFO] Base: {dataset} | construcao: {args.construction} | recompensa: {args.reward} | "
          f"saida: {results_dir}", flush=True)
    if dataset == "survey":
        print("[AVISO] A base survey tem grafos de ate ~534 mil vertices: cada execucao de 5.400 construcoes "
              "pode levar horas. Considere --instances e --seeds.", flush=True)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = configure_logger(os.path.join(LOGS_DIR, f"main_icits_{run_timestamp}.log"))
    runs_file = os.path.join(results_dir, "runs.csv")
    decisions_file = os.path.join(results_dir, "decisions.csv")
    instances_file = os.path.join(results_dir, "instances.csv")
    if args.fresh:
        for f in (runs_file, decisions_file, instances_file):
            if os.path.exists(f):
                os.remove(f)

    plan = build_plan(dataset, args.experiment, args.selectors, args.reward, args.instances, args.methods)
    instance_names = list(dict.fromkeys(i for i, _ in plan))
    jobs = pending_jobs(plan, args.seeds, results_dir, args.train, args.greedy, dataset, args.construction)
    total_planned = len(plan) * args.seeds
    logger.info("Base %s | construcao %s | metodos: %s", dataset, args.construction, sorted({m for _, m in plan}))

    workers, how = resolve_workers(args.workers, args.reserve_cores, max(len(jobs), len(instance_names)))
    logger.info("Paralelismo: %s", how)
    try:
        import psutil
        parent = (os.getpid(), psutil.Process(os.getpid()).create_time())
    except ImportError:  # sem psutil, os workers não vigiam o principal
        parent = (None, None)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                               initargs=parent)
    try:
        # ------------------------------------------------------------
        # ETAPA 1 (paralela): grafo, centralidades, B0, RCM e limite inferior de cada instância
        # ------------------------------------------------------------
        prepared, t0 = {}, time.time()
        futures = {pool.submit(_prepare_job, name, dataset, args.refresh_cache): name for name in instance_names}
        for f in as_completed(futures):
            name = futures[f]
            res = prepared[name] = f.result()
            info = res["info"]
            logger.info("Instancia %s: |V|=%s |E|=%s B0=%s RCM=%s LB=%s (%.1fs%s)", name, info["Nodes"],
                        info["Edges"], info["Initial Bandwidth"], info["RCM Bandwidth"], info["Lower Bound"],
                        res["elapsed"], f"; inicio uniforme para {res['invalid']}" if res["invalid"] else "")
        logger.info("Preparacao de %s instancia(s) em %.1fs", len(prepared), time.time() - t0)
        inst_df = pd.DataFrame([{**prepared[n]["info"], **{f"Time {c} (s)": t for c, t in prepared[n]["times"].items()},
                                 "Uniform Start Fallback": ",".join(prepared[n]["invalid"])} for n in instance_names])
        if os.path.exists(instances_file):  # mantém instâncias registradas por execuções anteriores na pasta
            old = pd.read_csv(instances_file)
            inst_df = pd.concat([old[~old["Instance"].isin(inst_df["Instance"])], inst_df], ignore_index=True)
        inst_df.to_csv(instances_file, index=False)

        # ------------------------------------------------------------
        # ETAPA 2 (paralela): uma tarefa por (instância, método, semente); as maiores primeiro, para que
        # nenhuma execução longa fique sozinha no fim enquanto os outros processos esperam
        # ------------------------------------------------------------
        logger.info("%s execucoes planejadas, %s ja feitas, %s a executar | orcamento: %s+%s decisoes x %s "
                    "construcoes | %s processo(s)", total_planned, total_planned - len(jobs), len(jobs),
                    args.train, args.greedy, N_CALLS * ITER_MAX, workers)
        jobs.sort(key=lambda j: prepared[j[0]]["info"]["Edges"], reverse=True)
        futures = {pool.submit(run_job, i, m, s, args.train, args.greedy, not args.no_decisions, dataset,
                               args.construction): (i, m, s) for i, m, s in jobs}
        t0 = time.time()
        for k, f in enumerate(as_completed(futures), start=1):
            name, method, seed = futures[f]
            summary, decision_rows = f.result()
            append_csv(decision_rows, decisions_file)   # decisões antes do resumo: runs.csv marca a execução
            append_csv([summary], runs_file)            # como concluída só depois que tudo foi gravado
            elapsed = time.time() - t0
            logger.info("[OK] %s/%s %s %s seed %s: B0 %s -> %s (RCM %s) em %.1fs | restante estimado %.0f min",
                        k, len(jobs), name, method, seed, summary["Initial Bandwidth"], summary["Best Bandwidth"],
                        summary["RCM Bandwidth"], summary["Global Time (s)"], elapsed / k * (len(jobs) - k) / 60)
    finally:
        pool.shutdown(cancel_futures=True)

    logger.info("Resultados: %s | decisoes: %s | instancias: %s", runs_file, decisions_file, instances_file)
    logger.info("Relatorio: python icits2026/report_icits.py --results-dir %s", results_dir)
    print("[INFO] Execucao finalizada com sucesso.", flush=True)
    return results_dir


def main(argv=None):
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
