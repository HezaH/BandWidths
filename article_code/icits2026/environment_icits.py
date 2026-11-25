# Ambiente do artigo ICITS 2026 (Secoes 3.1 e 3.2).
#
# Baseado em article_code/enviroment.py (Env do main_survey.py), com as mudancas do artigo:
#   - cada decisao executa N_CALLS x ITER_MAX = 15 construcoes com a centralidade escolhida e
#     devolve a melhor banda B_t (igual ao Env original);
#   - estado normalizado de 12 dimensoes:
#       s_t = [t/T, B_t/B0, B*_t/B0, (B0 - B*_t)/B0, tau_t/T, rho_1,t, ..., rho_7,t]
#     T = horizonte de treino, B0 = banda da ordenacao original, B_t = banda da ultima decisao,
#     B*_t = melhor banda ate aqui (incumbente), tau_t = decisoes desde a ultima melhoria estrita
#     e rho_i,t = fracao das escolhas da acao i que melhoraram estritamente o incumbente;
#   - recompensa densa (B*_{t-1} = incumbente antes da acao a_t):
#       (B*_{t-1} - B_t) / B0              se B_t <  B*_{t-1}
#       0.01                               se B_t == B*_{t-1}
#       -0.05 * min((B_t - B*_{t-1}) / B0, 1)  se B_t >  B*_{t-1}
#
# O incumbente comeca em B0 (a ordenacao do arquivo), como no artigo e no main_survey.py.
#
# Extensao E2 (conclusao do artigo: "rescale the reward so that every strict improvement outweighs
# a tie"): reward_scheme = "half_unit_tie" troca o premio do empate de 0.01 por 0.5 / B0. Como a
# banda e inteira, a menor melhoria possivel (1 unidade) rende 1 / B0, sempre o dobro do empate.
# Melhoria e piora ficam identicas as do artigo, entao qualquer diferenca vem so dessa troca.

import numpy as np

from centralities import centrality_heuristic

TIE_REWARD = 0.01
WORSE_PENALTY = 0.05
REWARD_SCHEMES = ("paper", "half_unit_tie")


class EnvICITS:
    def __init__(self, initial_bandwidth: float, n_actions: int, horizon: int, alpha: float = 0.3,
                 n_calls: int = 5, iter_max: int = 3, parent_order: bool = False, reward_scheme: str = "paper"):
        """
        initial_bandwidth -> B0, banda da ordenação original (incumbente inicial).
        n_actions         -> número de centralidades (tamanho de rho no estado).
        horizon           -> T, horizonte de treino (normaliza t e tau).
        alpha, n_calls, iter_max, parent_order -> parâmetros do construtivo, como no Env original.
                             parent_order=False reproduz o construtivo do artigo (cada nível inteiro
                             ordenado pela centralidade e embaralhado pela LCR). parent_order=True é a
                             extensão E1: os filhos de cada vértice são rotulados na ordem do pai
                             (regra de Cuthill-McKee), e a LCR ordena só os filhos de cada pai.
        reward_scheme     -> "paper" (Seção 3.2 do artigo) ou "half_unit_tie" (extensão E2).
        """
        if reward_scheme not in REWARD_SCHEMES:
            raise ValueError(f"reward_scheme inválido: {reward_scheme}. Use um de {REWARD_SCHEMES}")
        self.reward_scheme = reward_scheme
        self.B0 = float(initial_bandwidth)
        self.n_actions = n_actions
        self.horizon = horizon
        self.alpha = alpha
        self.n_calls = n_calls
        self.iter_max = iter_max
        self.parent_order = parent_order

        self.t = 0                                  # decisões já tomadas
        self.last_bandwidth = self.B0               # B_t (antes da 1ª decisão: B0)
        self.best_sol = self.B0                     # B*_t, incumbente
        self.best_solution = None                   # rotulação do incumbente
        self.since_improvement = 0                  # tau_t
        self.selections = np.zeros(n_actions)       # nº de escolhas de cada ação
        self.improvements = np.zeros(n_actions)     # nº de melhorias estritas por ação

    def success_rates(self) -> np.ndarray:
        """rho_i: fração das escolhas da ação i que melhoraram estritamente o incumbente."""
        return np.divide(self.improvements, self.selections, out=np.zeros(self.n_actions),
                         where=self.selections > 0)

    def get_state(self) -> list:
        B0, T = self.B0, float(self.horizon)
        return [self.t / T, self.last_bandwidth / B0, self.best_sol / B0, (B0 - self.best_sol) / B0,
                self.since_improvement / T, *self.success_rates().tolist()]

    def reward(self, bandwidth: float, previous_best: float) -> float:
        if bandwidth < previous_best:
            return (previous_best - bandwidth) / self.B0
        if bandwidth == previous_best:
            return 0.5 / self.B0 if self.reward_scheme == "half_unit_tie" else TIE_REWARD
        return -WORSE_PENALTY * min((bandwidth - previous_best) / self.B0, 1.0)

    def step(self, graph, action: int, centrality_values: dict, cent_str: str, centralities: dict,
             start_strategy: str = "centrality") -> dict:
        """Uma decisão: n_calls x iter_max construções com a centralidade da ação escolhida."""
        best_cost, best_solution, iterations = float("inf"), None, []
        for _ in range(self.n_calls):
            cost, _, solution, it = centrality_heuristic(
                graph=graph, centrality_values=centrality_values, cent_str=cent_str, alpha=self.alpha,
                iter_max=self.iter_max, centralities=centralities, parent_order=self.parent_order,
                start_strategy=start_strategy)
            iterations.extend(it)
            if cost < best_cost:
                best_cost, best_solution = cost, solution

        previous_best = self.best_sol
        reward = self.reward(best_cost, previous_best)
        self.selections[action] += 1
        improved = best_cost < previous_best
        if improved:
            self.best_sol, self.best_solution = best_cost, best_solution
            self.improvements[action] += 1
            self.since_improvement = 0
        else:
            self.since_improvement += 1
        self.last_bandwidth = best_cost
        self.t += 1

        return {"bandwidth": best_cost, "incumbent": self.best_sol, "reward": reward, "improved": improved,
                "step_solution": best_solution, "iterations": iterations}
