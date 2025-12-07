# Importações de bibliotecas padrão e locais
import math
import random
import numpy as np
import networkx as nx
from collections import deque, defaultdict
import os
from scipy.io import mmread
import itertools
from multiprocessing import Pool, cpu_count
import pandas as pd

# Importa funções auxiliares de outro arquivo no projeto
from aprendizado_reforco import (
    get_LCR,
    read_unweighted_graph,
    calc_bandwidth,
    reorder_graph,
    plot_graph,
    plot_graph_as_matrix
                                 )
                                 
# ---------------------------
# Limite dual (lower bound) e utilitários de recompensa
# ---------------------------
def lower_bound_bandwidth(n, m):
    """
    Calcula o limite inferior para a largura de banda (B_f(G)) de um grafo.
    Este é um valor teórico mínimo que a largura de banda pode atingir.
    A fórmula utilizada é: B_f(G) >= n ± (1 + sqrt((2n ± 1)^2 - 8m)) / 2
    A função testa as 4 combinações de sinais (±) e retorna o maior valor real obtido.
    
    Args:
        n (int): Número de vértices do grafo.
        m (int): Número de arestas do grafo.
        
    Returns:
        float: O valor do limite inferior da largura de banda.
    """
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
    """
    Calcula a recompensa para o agente de RL com base na qualidade da largura de banda (bw) obtida.
    A recompensa é o negativo do "gap", que é a diferença entre a largura de banda e o limite inferior.
    O objetivo é minimizar esse gap. A recompensa é normalizada pelo número de nós.
    
    Args:
        bw (int): Largura de banda calculada para uma dada ordenação.
        n (int): Número de vértices do grafo.
        m (int): Número de arestas do grafo.
        
    Returns:
        tuple: A recompensa normalizada (float) e o limite inferior (float).
    """
    lb = lower_bound_bandwidth(n, m)
    gap = max(0.0, bw - lb)
    # normalização simples para manter magnitudes estáveis
    denom = max(1.0, n) 
    return -gap / denom, lb

# ---------------------------
# Discretização de estado (leve, tabular)
# ---------------------------
def make_state(n, level_size, unvisited_frac, last_action, num_actions):
    """
    Cria uma representação de estado discreta para a Q-table.
    O estado é uma tupla que captura informações sobre o processo de busca:
      - Tamanho do nível atual na busca em largura (discretizado em bins).
      - Fração de nós ainda não visitados (discretizado em bins).
      - Ação (centralidade) escolhida no passo anterior.
      
    Args:
        level_size (int): O número de nós no nível atual da BFS.
        unvisited_frac (float): A fração de nós ainda não visitados.
        last_action (int): O índice da ação tomada no passo anterior.
        num_actions (int): O número total de ações possíveis.
        
    Returns:
        tuple: O estado discreto.
    """
    def bin_level(sz, n, bins=5):
        """
        Discretize the level size into bins proportional to graph size n.
        
        Args:
            sz (int): size of the level (number of nodes).
            n (int): total number of nodes in the graph.
            bins (int): number of bins (default=5).
        """
        if sz == 0:
            return 0
        
        # tamanho de cada faixa
        step = n / bins
        
        # calcula bin dinamicamente
        for i in range(1, bins+1):
            if sz <= i * step:
                return i
        return bins

    def bin_frac(f, bins=5):
        """
        Discretize the fraction of unvisited nodes into bins.
        
        Args:
            f (float): fraction of unvisited nodes (0 <= f <= 1).
            bins (int): number of bins (default=5).
        """
        step = 1.0 / bins
        for i in range(bins):
            if f <= (i+1) * step:
                return i
        return bins-1

    # O estado é a combinação dos valores discretizados e da última ação.
    return (bin_level(level_size, n), bin_frac(unvisited_frac), last_action if last_action is not None else num_actions)

# ---------------------------
# Política ε-greedy sobre Q-table
# ---------------------------
class QPolicy:
    """
    Implementa uma política de Q-learning com uma estratégia ε-greedy.
    A Q-table armazena os valores de ação-estado (Q(s,a)).
    """
    def __init__(self, num_actions, lr=0.001, gamma=0.9, eps=1.0, eps_min=0.1, eps_decay=0.995):
        """
        Inicializa a política.
        
        Args:
            num_actions (int): Número de ações possíveis (neste caso, o número de centralidades).
            lr (float): Taxa de aprendizado (learning rate).
            gamma (float): Fator de desconto para recompensas futuras.
            eps (float): Probabilidade inicial de exploração (escolher ação aleatória).
            eps_min (float): Valor mínimo de epsilon.
            eps_decay (float): Fator de decaimento de epsilon a cada episódio.
        """
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=float))

    def select_action(self, state):
        """
        Seleciona uma ação para um dado estado usando a política ε-greedy.
        Com probabilidade ε, escolhe uma ação aleatória (exploração).
        Com probabilidade 1-ε, escolhe a ação com o maior valor Q (explotação).
        """
        if random.random() < self.eps:
            return random.randrange(self.num_actions)
        qvals = self.Q[state]
        # desempate aleatório entre máximos
        maxq = np.max(qvals)
        best = [a for a, q in enumerate(qvals) if q == maxq]
        return random.choice(best)

    def update(self, s, a, r, s_next, done):
        """
        Atualiza o valor Q para um par estado-ação usando a equação de Bellman.
        Q(s,a) <- Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))
        
        Args:
            s: O estado atual.
            a: A ação tomada.
            r: A recompensa recebida.
            s_next: O próximo estado.
            done (bool): Se o episódio terminou.
        """
        qsa = self.Q[s][a]
        target = r
        if not done:
            target += self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] = qsa + self.lr * (target - qsa)

    def end_episode(self):
        """
        Atualiza o valor de epsilon no final de cada episódio para reduzir a exploração ao longo do tempo.
        """
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

# ---------------------------
# BFS-LCR dirigido por política (ação = qual centralidade usar no nível)
# ---------------------------
def bfs_LCR_with_policy(G, centralities_list, policy):
    """
    Executa uma busca em largura (BFS) modificada, onde a política de RL decide qual
    métrica de centralidade usar para ordenar os nós em cada nível da busca.
    A ordenação é feita usando uma Lista de Candidatos Restrita (LCR).
    
    Args:
        G (nx.Graph): O grafo a ser reordenado.
        centralities_list (list): Lista de vetores de centralidade.
        policy (QPolicy): A política de RL a ser usada.
        
    Returns:
        tuple: A nova ordem dos nós, a largura de banda, o limite inferior e a recompensa final.
    """
    n = G.number_of_nodes()
    visited = set()
    order = []
    node_list = list(G.nodes())

    # Para estado
    last_action = None

    # A busca continua até que todos os nós tenham sido visitados
    while len(visited) < n:
        # Escolhe um nó inicial aleatório que ainda não foi visitado
        start_node = random.choice([v for v in node_list if v not in visited])
        visited.add(start_node)
        order.append(start_node)

        queue = deque([start_node])
        while queue:
            # O nível atual consiste em todos os nós na fila
            current_level = list(queue)
            queue.clear()

            # Coleta vizinhos não visitados dos nós no nível atual
            neighbors = {nb for node in current_level for nb in G.neighbors(node) if nb not in visited}

            # Cria o estado atual para a política de RL
            level_size = len(current_level)
            unvisited_frac = (n - len(visited)) / n
            s = make_state(n, level_size, unvisited_frac, last_action, len(centralities_list))

            # A política seleciona uma ação (qual centralidade usar)
            a = policy.select_action(s)
            cent_vec = centralities_list[a]

            # Constrói a Lista de Candidatos Restrita (LCR)
            # O limiar alpha é adaptativo, influenciado por epsilon, para controlar a ganância
            alpha = policy.eps
            lcr = get_LCR(neighbors, cent_vec, alpha)

            # Adiciona os nós da LCR à ordem e à fila para o próximo nível
            for v in lcr:
                if v not in visited:
                    visited.add(v)
                    order.append(v)
                    queue.append(v)

            # Prepara o próximo estado
            level_size_next = len(queue)
            unvisited_frac_next = (n - len(visited)) / n
            s_next = make_state(n, level_size_next, unvisited_frac_next, a, len(centralities_list))

            # A recompensa intermediária é 0. A recompensa real é dada apenas no final do episódio.
            r = 0.0
            policy.update(s, a, r, s_next, done=False)
            last_action = a

    # Após o término da busca, calcula a recompensa final com base na largura de banda total
    bw = calc_bandwidth(G, order)
    m = G.number_of_edges()
    r_final, lb = reward_from_bandwidth(bw, n, m)

    # Faz a atualização final da Q-table com a recompensa do episódio completo
    terminal_state = make_state(n, 0, 0.0, last_action, len(centralities_list))
    policy.update(terminal_state, last_action, r_final, terminal_state, done=True)

    return order, bw, lb, r_final

# ---------------------------
# Treino e uso (Q-MCH)
# ---------------------------
def q_mch_train(graphs, centralities_by_graph, episodes=30,
                lr=0.001, gamma=0.9, eps=1.0, eps_min=0.1, eps_decay=0.995):
    """
    Treina a política de Q-learning em um conjunto de grafos.
    
    Args:
        graphs (list): Lista de grafos (objetos NetworkX) para treinamento.
        centralities_by_graph (list): Lista contendo as centralidades para cada grafo.
        episodes (int): O número de episódios de treinamento.
        ...outros hiperparâmetros de RL.
        
    Returns:
        QPolicy: A política treinada.
    """
    # O número de ações é o número de métricas de centralidade disponíveis.
    num_actions = len(centralities_by_graph[0])
    policy = QPolicy(num_actions, lr=lr, gamma=gamma, eps=eps, eps_min=eps_min, eps_decay=eps_decay)

    for ep in range(episodes):
        # Embaralha as instâncias a cada episódio para variar a ordem de treinamento
        idxs = list(range(len(graphs)))
        random.shuffle(idxs)

        ep_rewards = []
        for gi in idxs:
            G = graphs[gi]
            centrals = centralities_by_graph[gi]
            # Executa um episódio completo (uma reordenação) e obtém a recompensa
            order, bw, lb, r_final = bfs_LCR_with_policy(G, centrals, policy)
            ep_rewards.append(r_final)

        # Ao final do episódio, decai o epsilon e imprime o progresso
        policy.end_episode()
        print(f"[Treino] Episódio {ep+1}/{episodes} | recompensa média {np.mean(ep_rewards):.4f} | eps={policy.eps:.3f}")

    return policy

def q_mch_solve(G, centralities_list, policy):
    """
    Usa a política já treinada para encontrar uma boa ordenação para um grafo.
    A exploração (epsilon) é desativada para que a política sempre escolha a melhor ação conhecida.
    
    Args:
        G (nx.Graph): O grafo a ser resolvido.
        centralities_list (list): A lista de centralidades para o grafo.
        policy (QPolicy): A política treinada.
        
    Returns:
        tuple: A ordem encontrada, a largura de banda e o limite inferior.
    """
    old_eps = policy.eps
    policy.eps = 0.0  # Desliga a exploração (modo greedy)
    order, bw, lb, r_final = bfs_LCR_with_policy(G, centralities_list, policy)
    policy.eps = old_eps # Restaura o valor original de epsilon
    return order, bw, lb

# 1) Pré-cálculo de centralidades indexadas por nó:
#    (mantive degree/closeness/betweenness como no seu código — se estiver lento, troque por versões aproximadas)

def fast_centralities(G, centralities: dict):
    centralities_list = []

    for name, func in centralities.items():
        if name in ["Eigenvector"]:
            result = func(G, max_iter=1000)
        # Some functions require parameters (like Katz centrality)
        elif name == "Katz Centrality":
            result = func(G, alpha=0.005, beta=1.0, max_iter=2000)
        elif name == "PageRank":
            result = func(G, alpha=0.85)
        else:
            result = func(G)
        print(f"Calculing for: {name}")
        cent_dict = result
        # Convert the dict to a vector aligned with node ordering
        cent_vector = np.array([cent_dict[v] for v in G.nodes()], dtype=float)
        centralities_list.append(cent_vector)

    return centralities_list

# ==== Função de treino + avaliação para grid search ====
def run_case(args):
    A_local, centralities_local, params = args
    episodes, lr, gamma, eps, eps_decay = params
    
    policy = q_mch_train(
        graphs=[A_local],
        centralities_by_graph=[centralities_local],
        episodes=episodes, lr=lr, gamma=gamma, eps=eps_decay
    )

    order, bw, lb = q_mch_solve(A_local, centralities_local, policy)
    return (params, bw, lb, order)  # <<< apenas dados simples

# ==== Bloco principal ====
if __name__ == "__main__":
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix')
    mtx_files = [f for f in os.listdir(log_folder) if f.endswith('.mtx')]

    # Lista para armazenar resultados
    resultados = []

    for arquivo in mtx_files:
        print(f"\n=== Processando {arquivo} ===")
        file_path = os.path.join(log_folder, arquivo)

        # Leitura e cálculo de centralidades
        A = read_unweighted_graph(file_path)
        dict_centralities = {
        #Standard centrality measures
        "Degree": nx.degree_centrality,
        "Closeness": nx.closeness_centrality,
        "Betweenness": nx.betweenness_centrality,
        "Eigenvector": nx.eigenvector_centrality,
        
        # Additional centrality measures
        "Katz Centrality": nx.katz_centrality,
        "PageRank": nx.pagerank,
        "Harmonic Centrality": nx.harmonic_centrality,
        "Current-flow Betweenness": nx.current_flow_betweenness_centrality
        }
        centralities_list = fast_centralities(A, dict_centralities)

        # Largura de banda do grafo original
        bw_original = calc_bandwidth(A, list(A.nodes()))

        # Grid de parâmetros
        episodes_vals   = [20, 30, 40]
        lr_vals         = [0.001, 0.005]
        gamma_vals      = [0.8, 0.9]
        eps_vals        = [1.0, 0.5]
        eps_decay_vals  = [0.99, 0.995]
        grid_params     = list(itertools.product(episodes_vals, lr_vals, gamma_vals, eps_vals, eps_decay_vals))
        grid_args       = [(A, centralities_list, params) for params in grid_params]

        # Execução em paralelo
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(run_case, grid_args)

        # Ordena pela melhor largura de banda
        results_sorted = sorted(results, key=lambda x: x[1])

        print("\nTop 3 parâmetros:")
        for params, bw, lb, _ in results_sorted[:3]:
            print(f"Params: {params} | BW: {bw} | LB: {lb} | GAP: {bw - lb:.2f}")

        # Pega melhor configuração
        best_params, _, _, best_order = results_sorted[0]

        # Re-treina no main process para ter o policy
        best_policy = q_mch_train(
            graphs=[A],
            centralities_by_graph=[centralities_list],
            episodes=best_params[0],
            lr=best_params[1],
            gamma=best_params[2],
            eps=best_params[3],
            eps_decay=best_params[4]
        )

        # Aplica política
        best_order, best_bw, best_lb = q_mch_solve(A, centralities_list, best_policy)

        # Resultados finais
        print("\n=== Resultado com melhor configuração ===")
        print(f"Melhores parâmetros: {best_params}")
        print(f"Largura de banda: {best_bw} | LB: {best_lb:.2f} | GAP: {best_bw - best_lb:.2f}")

        # Adiciona ao comparativo
        resultados.append({
            'arquivo': arquivo,
            'bw_original': bw_original,
            'bw_qlearning': best_bw,
            'gap': bw_original - best_bw
        })

        # Plots (apenas no main thread)
        file_name = os.path.splitext(arquivo)[0]
        plot_graph(A, file_name)
        plot_graph_as_matrix(A, file_name + "_matrix")
        A_new = reorder_graph(A, best_order)
        plot_graph(A_new, file_name + "_q_learning")
        plot_graph_as_matrix(A_new, file_name + "_q_learning_matrix")
        A_new = A_new

    # Cria e exibe o DataFrame comparativo
    df_resultados = pd.DataFrame(resultados)
    print("\nComparativo de larguras de banda:")
    print(df_resultados)