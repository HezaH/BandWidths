import os
import math
import random
from collections import deque, defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from scipy.io import mmread
import networkx as nx

# ---------------------------
# Funções originais
# ---------------------------

def lower_bound_bandwidth(n, m):
    cands = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            disc = (2*n + s2*1)**2 - 8*m
            if disc >= 0:
                val = n + s1 * (1 + math.sqrt(disc)) / 2.0
                cands.append(val)
    if not cands:
        return 0.0
    return max(cands)

def reward_from_bandwidth(bw, n, m):
    lb = lower_bound_bandwidth(n, m)
    gap = max(0.0, bw - lb)
    denom = max(1.0, n) 
    return -gap / denom, lb

def make_state(level_size, unvisited_frac, last_action, num_actions):
    def bin_level(sz):
        if sz == 0:   return 0
        if sz <= 2:   return 1
        if sz <= 5:   return 2
        if sz <= 10:  return 3
        if sz <= 20:  return 4
        return 5
    def bin_frac(f):
        if f <= 0.1:  return 0
        if f <= 0.25: return 1
        if f <= 0.5:  return 2
        if f <= 0.75: return 3
        return 4
    return (bin_level(level_size), bin_frac(unvisited_frac), last_action if last_action is not None else num_actions)

def calc_bandwidth(G, order):
    pos = {node: i for i, node in enumerate(order)}
    return max(abs(pos[u] - pos[v]) for u, v in G.edges())

def get_LCR(neighbors, cent_vec, alpha):
    if not neighbors:
        return []
    scores = [(v, cent_vec[v]) for v in neighbors]
    scores.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(scores) * alpha))
    return [v for v, _ in scores[:k]]

def reorder_graph(G, order):
    mapping = {node: i for i, node in enumerate(order)}
    return nx.relabel_nodes(G, mapping)

def plot_graph(G, title):
    plt.figure(figsize=(4, 4))
    nx.draw(G, node_size=50)
    plt.title(title)
    plt.show()

def read_unweighted_graph(file_path):
    if file_path.endswith(".mtx"):
        # Lê a matriz esparsa
        from scipy.io import mmread
        mat = mmread(file_path).tocoo()
        # Converte para grafo não direcionado, sem pesos
        G = nx.Graph()
        G.add_edges_from(zip(mat.row, mat.col))
        return G
    elif file_path.endswith(".adjlist"):
        return nx.read_adjlist(file_path)
    else:
        return nx.read_edgelist(file_path)
# ---------------------------
# Centralidades rápidas
# ---------------------------
def fast_centralities(G, k_bet=50, k_clo=50, seed=42):
    random.seed(seed)
    n = len(G)

    deg = [deg_val / (n - 1) for _, deg_val in G.degree()]
    sample_nodes_clo = random.sample(list(G.nodes()), min(k_clo, n))
    clo_dict = {v: nx.closeness_centrality(G, u=v) if v in sample_nodes_clo else 0 for v in G.nodes()}
    clo = [clo_dict[v] for v in G.nodes()]
    bet_dict = nx.betweenness_centrality(G, k=min(k_bet, n), normalized=True, seed=seed)
    bet = [bet_dict[v] for v in G.nodes()]

    return [deg, clo, bet]

# ---------------------------
# Ambiente Gym
# ---------------------------
class BandwidthEnv(gym.Env):
    metadata = {}
    def __init__(self, G, centralities_list, alpha=0.5, seed=42):
        super().__init__()
        self.G = G
        self.centralities_list = centralities_list
        self.num_actions = len(centralities_list)
        self.alpha = alpha
        self.observation_space = spaces.MultiDiscrete([6, 5, self.num_actions + 1])
        self.action_space = spaces.Discrete(self.num_actions)
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _obs(self):
        level_size = len(self.queue)
        unvisited_frac = (self.n - len(self.visited)) / self.n
        return np.array(make_state(level_size, unvisited_frac, self.last_action, self.num_actions), dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.n = self.G.number_of_nodes()
        self.visited = set()
        self.order = []
        self.node_list = list(self.G.nodes())
        self.last_action = None
        start_node = self._rng.choice(self.node_list)
        self.visited.add(start_node)
        self.order.append(start_node)
        self.queue = deque([start_node])
        return self._obs(), {}

    def step(self, action):
        current_level = list(self.queue)
        self.queue.clear()
        neighbors = {nb for node in current_level for nb in self.G.neighbors(node) if nb not in self.visited}
        cent_vec = self.centralities_list[action]
        lcr = get_LCR(neighbors, cent_vec, self.alpha)
        for v in lcr:
            if v not in self.visited:
                self.visited.add(v)
                self.order.append(v)
                self.queue.append(v)
        done = len(self.visited) == self.n
        reward = 0.0
        info = {}
        if done:
            bw = calc_bandwidth(self.G, self.order)
            m = self.G.number_of_edges()
            reward, lb = reward_from_bandwidth(bw, self.n, m)
            info.update(dict(bw=bw, lb=lb, order=self.order))
        self.last_action = action
        return self._obs(), reward, done, False, info


def plot_graph_as_matrix(G, title="Matriz de Adjacência"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Gera matriz de adjacência como array NumPy (ordem natural dos nós no G)
    A = nx.to_numpy_array(G, dtype=int)

    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(A, cmap="Greys")  # tons de cinza: 1 = aresta, 0 = sem aresta

    # Ajustes de eixo
    ax.set_title(title, pad=15)
    ax.set_xlabel("Nó j")
    ax.set_ylabel("Nó i")

    # Opcional: ticks para alguns nós apenas (evita poluição visual)
    n = A.shape[0]
    step = max(1, n // 10)
    ax.set_xticks(range(0, n, step))
    ax.set_yticks(range(0, n, step))

    # Barra de cor
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    plt.show()


# ---------------------------
# Execução principal
# ---------------------------
if __name__ == "__main__":
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix')
    mtx_files = [f for f in os.listdir(log_folder) if f.endswith('.mtx')]

    for arquivo in mtx_files:
        print(f"Arquivo encontrado: {arquivo}")
        file_path = os.path.join(log_folder, arquivo)
        
        A = read_unweighted_graph(file_path)
        plot_graph_as_matrix(A, arquivo)
        
        centralities_list = fast_centralities(A)

        env = BandwidthEnv(A, centralities_list)
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            gamma=0.9,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            exploration_fraction=0.7,
            verbose=1,
        )
        print("Treinando...")
        model.learn(total_timesteps=10_000)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        print(f"Resultado final: BW={info['bw']} | LB={info['lb']:.2f} | Gap={info['bw'] - info['lb']:.2f}")
        A_new = reorder_graph(A, info['order'])
        plot_graph_as_matrix(A_new, arquivo + "_q_learning")
