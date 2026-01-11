import copy
import random
import numpy as np
from centralities import centrality_heuristic

class Env:
    def __init__(self, opt:float ):
        self.opt = opt
        self.n_steps = 0
        self.best_sol = opt
        self.gap = (self.best_sol - self.opt) / self.opt
        self.previous_cost = np.inf
        

    def reset(self):
        pass

    def step(self, graph, centrality_values, cent_str, centralities):
        melhor_custo = float('inf')
        k = 0
        iterations = []
        for _ in range(5):
            custo_s, graph_rebuilt, solution, it = centrality_heuristic(graph=graph, centrality_values=centrality_values, cent_str=cent_str, alpha=0.3, iter_max=3, centralities=centralities)
            
            iterations.extend(it)
            
            if custo_s < melhor_custo:
                melhor_custo = custo_s
                k = 0
            else:
                k += 1
        
        if melhor_custo < self.best_sol:
            self.best_sol = melhor_custo
            self.best_graph = graph_rebuilt
            self.best_solution = solution
        
        self.n_steps+=1
        if self.opt in [0, None]:
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