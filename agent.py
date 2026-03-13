import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from dqn import DQN


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.long),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int = 104,
        action_size: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        buffer_capacity: int = 10_000,
        min_buffer: int = 500,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_buffer = min_buffer

        self.policy_net = DQN(state_size, 512, 256, 128, action_size)
        self.target_net = DQN(state_size, 512, 256, 128, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.train_step_count: int = 0
        self.target_update_freq = target_update_freq

        # Metrics
        self.episode_count: int = 0
        self.score_history: list = []

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.min_buffer:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self._update_target()

        return loss.item()

    def _update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def record_score(self, score: int):
        self.score_history.append(score)
        if len(self.score_history) > 100:
            self.score_history.pop(0)
        self.episode_count += 1

    def mean_score(self) -> float:
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)

    def get_metrics(self) -> dict:
        return {
            "episode":    self.episode_count,
            "epsilon":    round(self.epsilon, 4),
            "mean_score": round(self.mean_score(), 2),
            "buffer_size": len(self.buffer),
        }
