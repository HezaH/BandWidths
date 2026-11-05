import copy
import random
import numpy as np
from centralities import centrality_heuristic

class Env:
    def __init__(self, opt:float, lower_bound:float=None, alpha:float=0.3, n_calls:int=5, iter_max:int=3, parent_order:bool=True):
        """
        opt         -> valor de referência inicial (best_sol começa nele).
        lower_bound -> limite dual (ex.: Lam et al. 1997, como no artigo SBPO). Quando
                       informado, gap = (custo - lower_bound) / custo, em [0, 1): 0 indica
                       que o limite foi atingido (ótimo). Sem ele, mantém o gap original
                       (opt - custo) / opt.
        alpha, n_calls, iter_max -> parâmetros da LCR e do multi-start de cada passo
                       (n_calls chamadas de centrality_heuristic com iter_max construções).
        """
        self.opt = opt
        self.lower_bound = lower_bound
        self.alpha = alpha
        self.n_calls = n_calls
        self.iter_max = iter_max
        self.parent_order = parent_order
        self.n_steps = 0
        self.best_sol = opt
        self.best_graph = None
        self.best_solution = None
        self.gap = self._gap(self.best_sol) if lower_bound is not None else 0.0
        self.previous_cost = np.inf


    def reset(self):
        pass

    def _gap(self, cost: float) -> float:
        """Gap relativo ao limite dual: (custo - limite) / custo."""
        return (cost - self.lower_bound) / float(cost) if cost > 0 else 0.0

    def step(self, graph, centrality_values, cent_str, centralities, alpha=None, start_strategy="centrality", local_search=False):
        """
        Um passo do ambiente: n_calls chamadas de centrality_heuristic com a ação escolhida.
        alpha=None usa o alpha do ambiente; start_strategy e local_search são repassados ao
        construtivo (permitem que a ação do agente seja o alpha ou a estratégia de início).
        """
        alpha = self.alpha if alpha is None else alpha
        melhor_custo = float('inf')
        melhor_graph, melhor_solution = None, None
        k = 0
        iterations = []
        for _ in range(self.n_calls):
            custo_s, graph_rebuilt, solution, it = centrality_heuristic(graph=graph, centrality_values=centrality_values, cent_str=cent_str, alpha=alpha, iter_max=self.iter_max, centralities=centralities, parent_order=self.parent_order, start_strategy=start_strategy, local_search=local_search)

            iterations.extend(it)

            if custo_s < melhor_custo:
                # guarda a solução da MELHOR chamada (antes ficava a da última chamada,
                # que podia não corresponder a melhor_custo)
                melhor_custo = custo_s
                melhor_graph, melhor_solution = graph_rebuilt, solution
                k = 0
            else:
                k += 1

        if melhor_custo < self.best_sol:
            self.best_sol = melhor_custo
            self.best_graph = melhor_graph
            self.best_solution = melhor_solution

        self.n_steps+=1
        if self.lower_bound is not None:
            self.gap = self._gap(melhor_custo)
        elif self.opt in [0, None]:
            self.opt = melhor_custo
            self.gap = 0
        else:
            self.gap = (self.opt - melhor_custo) / float(self.opt)
        reward = self.reward(melhor_custo)
        self.previous_cost = melhor_custo


        return {"n_steps": self.n_steps,
                "gap": self.gap,
                "reward": reward,
                "centrality":cent_str ,
                "bandwidth":melhor_custo,
                "graph": self.best_graph,
                "solution": self.best_solution,
                "step_solution": melhor_solution,
                "iterations": iterations
                }


    def reward(self, current_cost: float):
        if (current_cost <= self.best_sol) and (current_cost <= self.previous_cost):
            return 10
        elif current_cost <= self.previous_cost:
            return 5
        elif (current_cost > self.best_sol) or (current_cost > self.previous_cost):
            return -2


    def get_initial_state(self):
        return [self.n_steps, self.gap, 0, self.best_sol]
