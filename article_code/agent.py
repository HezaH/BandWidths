from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor as T
from itertools import permutations
from torch.nn import Module, Linear, MSELoss
import torch.nn.functional as F
import torch
from torch.optim import Adam

@dataclass
class Agent:
    learning_rate: float
    gamma: float
    n_movements: int
    n_actions: int
    n_states: int
    epsilon: float
    eps_min: float
    eps_dec: float
    Q: dict = field(default_factory=dict)
    deep: bool = True
    model_path: str = None


    def __post_init__(self):
        self.init_Q()
        self.action_space = list(range(self.n_movements))
        self.n_actions = len(self.action_space)

    def init_Q(self):
        if self.deep:
            self.Q = LinearQNet(self.learning_rate, self.n_states, self.n_actions)
            if self.model_path is not None:
                self.Q.load_state_dict(torch.load(self.model_path))
                self.Q.eval()
        else:
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q[(state, action)] = 0.0

    def choose_action(self, state, deep=True):
        action = None
        if deep:
            if np.random.random() > self.epsilon:
                state = torch.tensor(state, dtype=torch.float).to(self.Q.device)
                actions = self.Q.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            if np.random.random() < self.epsilon:
                action = np.random.choice([i for i in range(self.n_actions)])
            else:
                actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
                action = np.argmax(actions)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, next_state):
        if self.deep:
            self.Q.optimizer.zero_grad()
            states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
            actions = torch.tensor(action, dtype=torch.int).to(self.Q.device)
            rewards = torch.tensor(reward, dtype=torch.float).to(self.Q.device)
            next_states = torch.tensor(next_state, dtype=torch.float).to(self.Q.device)

            q_pred = self.Q.forward(states)[actions]
            q_next = self.Q.forward(next_states).max()

            q_target = rewards + self.gamma * q_next

            loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
            loss.backward()
            self.Q.optimizer.step()
        else:
            actions = np.array([self.Q[(next_state, a)] for a in range(self.n_actions)])
            a_max = np.argmax(actions)

            self.Q[(state, action)] += self.learning_rate * (reward +
                                                              self.gamma * self.Q[(next_state, a_max)] -
                                                              self.Q[(state, action)])

        self.decrement_epsilon()

    def get_movements(self):
        return self.action_space


class LinearQNet(Module):

    def __init__(self, learning_rate, input_dims, n_actions):
        super(LinearQNet, self).__init__()
        self.fc1 = Linear(input_dims, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, n_actions)

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss = MSELoss()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
