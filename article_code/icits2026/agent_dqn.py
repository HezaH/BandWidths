# Agente DQN do artigo ICITS 2026 (Secao 3.2, "State, Reward and Training").
#
# Diferencas em relacao a article_code/agent.py (usado pelo main_survey.py):
#   - rede online + rede alvo (target network) sincronizada a cada TARGET_SYNC atualizacoes;
#   - memoria de replay (capacidade 10.000) com mini-lotes de 64, aprendizado a partir de 64 transicoes;
#   - perda Smooth L1, Adam (lr = 0.001), gamma = 0.9 e norma do gradiente limitada a 1.0;
#   - alvo y = r + gamma * (1 - d) * max_a Q_alvo(s', a), com d marcando a transicao terminal;
#   - epsilon exponencial: eps_t = max(eps_min, eps_0 * (eps_min / eps_0) ** (t / T)).
#
# A arquitetura (duas camadas ocultas de 128 ReLU e uma saida por centralidade) e a mesma da
# LinearQNet de agent.py.

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """Duas camadas ocultas de 128 unidades ReLU e uma saída por ação (centralidade)."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Memória de replay circular: guarda (s, a, r, s', d) e sorteia mini-lotes uniformes."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.memory = deque(maxlen=capacity)
        self.rng = rng

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        idx = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.memory)


@dataclass
class DQNAgent:
    """
    Agente DQN com epsilon-greedy, replay e rede alvo (parâmetros da Seção 3.2 do artigo).

    n_states     -> dimensão do estado (12: 5 atributos de progresso + 7 taxas de sucesso).
    n_actions    -> número de centralidades (7).
    horizon      -> T, horizonte de treino (300 decisões) usado no decaimento de epsilon.
    seed         -> semente do gerador do agente (exploração e sorteio dos mini-lotes). O agente
                    usa um gerador próprio para não consumir o `random` usado pelo construtivo.
    """
    n_states: int
    n_actions: int
    horizon: int = 300
    learning_rate: float = 0.001
    gamma: float = 0.9
    eps_0: float = 1.0
    eps_min: float = 0.01
    buffer_capacity: int = 10_000
    batch_size: int = 64
    learn_start: int = 64
    target_sync: int = 25
    grad_clip: float = 1.0
    seed: int = 0
    n_updates: int = field(default=0, init=False)
    n_syncs: int = field(default=0, init=False)
    epsilon: float = field(default=1.0, init=False)

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.Q = QNetwork(self.n_states, self.n_actions)
        self.Q_target = QNetwork(self.n_states, self.n_actions)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(self.buffer_capacity, self.rng)
        self.epsilon = self.eps_0

    def epsilon_at(self, t: int) -> float:
        """eps_t = max(eps_min, eps_0 * (eps_min / eps_0) ** (t / T))."""
        return max(self.eps_min, self.eps_0 * (self.eps_min / self.eps_0) ** (t / self.horizon))

    def choose_action(self, state, t: int, greedy: bool = False) -> int:
        """Escolhe a ação por epsilon-greedy (greedy=True: fase gulosa, epsilon = 0)."""
        self.epsilon = 0.0 if greedy else self.epsilon_at(t)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q_values = self.Q(torch.tensor(state, dtype=torch.float32))
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Uma atualização com um mini-lote, a partir de `learn_start` transições na memória.
        Devolve a perda (ou None quando ainda não há transições suficientes)."""
        if len(self.memory) < max(self.learn_start, self.batch_size):
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_pred = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.Q_target(next_states).max(dim=1).values
            q_target = rewards + self.gamma * (1.0 - dones) * q_next
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), self.grad_clip)
        self.optimizer.step()
        self.n_updates += 1
        if self.n_updates % self.target_sync == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.n_syncs += 1
        return float(loss.item())
