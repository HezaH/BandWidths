# Seletores de bandit para a extensao E3 do artigo ICITS 2026.
#
# A Secao 4.4 do artigo observa que, como as acoes nao alteram o grafo, a formulacao e proxima de
# um bandit contextual, e aponta UCB1 (Auer et al., 2002) e adaptive pursuit (Thierens, 2005;
# Fialho et al., 2010) como comparadores naturais nao testados. Os dois seletores abaixo usam:
#   - a MESMA recompensa do DQN (a do esquema escolhido em EnvICITS), para que a comparacao isole o
#     mecanismo de aprendizado ("mesma recompensa, aprendiz diferente");
#   - a MESMA divisao de fases do DQN: aprendem nas `horizon` primeiras decisoes e, na fase gulosa,
#     escolhem sempre a acao de maior estimativa, sem explorar e sem atualizar.
#
# A recompensa do artigo vive numa escala pequena e variavel por instancia (empate = 0.01, melhorias
# de ~1/B0). UCB1 supoe recompensas em [0, 1]; por isso as recompensas sao normalizadas online pelo
# minimo e maximo ja observados (RunningMinMax), o que torna os dois seletores independentes da escala.

import math

import numpy as np


class RunningMinMax:
    """Normaliza valores para [0, 1] pelo mínimo e máximo observados até o momento."""

    def __init__(self):
        self.low, self.high = math.inf, -math.inf

    def __call__(self, value: float) -> float:
        self.low, self.high = min(self.low, value), max(self.high, value)
        return 0.5 if self.high == self.low else (value - self.low) / (self.high - self.low)


class UCB1Selector:
    """UCB1 (Auer et al., 2002): joga cada ação uma vez e depois escolhe
    argmax_a  média_a + c * sqrt(2 ln n / n_a)."""

    def __init__(self, n_actions: int, rng: np.random.Generator, c: float = 1.0):
        self.n_actions, self.rng, self.c = n_actions, rng, c
        self.counts = np.zeros(n_actions)
        self.sums = np.zeros(n_actions)
        self.normalize = RunningMinMax()

    def means(self):
        return np.divide(self.sums, self.counts, out=np.zeros(self.n_actions), where=self.counts > 0)

    def choose(self, greedy: bool = False) -> int:
        if greedy:
            return int(self.rng.choice(np.flatnonzero(self.means() == self.means().max())))
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(self.rng.choice(untried))
        n = self.counts.sum()
        ucb = self.means() + self.c * np.sqrt(2.0 * np.log(n) / self.counts)
        return int(self.rng.choice(np.flatnonzero(ucb == ucb.max())))

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        self.sums[action] += self.normalize(reward)


class AdaptivePursuitSelector:
    """Adaptive pursuit (Thierens, 2005): mantém uma estimativa Q_a por ação e um vetor de
    probabilidades que 'persegue' a melhor ação:
        Q_a   <- Q_a + alpha * (r - Q_a)                      (só a ação escolhida)
        P_a*  <- P_a* + beta * (P_max - P_a*)                 (a* = argmax Q)
        P_a   <- P_a  + beta * (P_min - P_a)   para a != a*
    com P_max = 1 - (K - 1) * P_min, de modo que toda ação mantém probabilidade mínima P_min."""

    def __init__(self, n_actions: int, rng: np.random.Generator, alpha: float = 0.8, beta: float = 0.8,
                 p_min: float = None):
        self.n_actions, self.rng, self.alpha, self.beta = n_actions, rng, alpha, beta
        self.p_min = p_min if p_min is not None else 0.1 / n_actions
        self.p_max = 1.0 - (n_actions - 1) * self.p_min
        self.q = np.full(n_actions, 0.5)
        self.p = np.full(n_actions, 1.0 / n_actions)
        self.normalize = RunningMinMax()

    def choose(self, greedy: bool = False) -> int:
        if greedy:
            return int(self.rng.choice(np.flatnonzero(self.q == self.q.max())))
        return int(self.rng.choice(self.n_actions, p=self.p / self.p.sum()))

    def update(self, action: int, reward: float):
        self.q[action] += self.alpha * (self.normalize(reward) - self.q[action])
        best = int(self.rng.choice(np.flatnonzero(self.q == self.q.max())))
        target = np.full(self.n_actions, self.p_min)
        target[best] = self.p_max
        self.p += self.beta * (target - self.p)
