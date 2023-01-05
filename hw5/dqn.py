import torch
import torch.nn as nn
import numpy as np

import random


class DQN:
    def __init__(self, model: nn.Module, state_dim: int, action_n: int, batch_size, gamma, lr, trajectory_n):
        self.action_n = action_n
        # self.model = model
        self.epsilon = 1
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        self.model = model
        # self.model = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.action_n)
        # )
        self.epsilon_decrease = 1 / trajectory_n
        self.memory = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def get_action(self, state):
        qvalues = self.model(torch.FloatTensor(state)).detach().numpy()
        prob = np.ones(self.action_n) * self.epsilon / self.action_n
        argmax_action = np.argmax(qvalues)
        prob[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return action

    def get_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in range(len(batch)):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            dones.append(batch[i][3])
            next_states.append(batch[i][4])
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        return states, actions, rewards, dones, next_states

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size:
            states, actions, rewards, dones, next_states = self.get_batch()

            qvalues = self.model(states)
            next_qvalues = self.model(next_states)

            targets = qvalues.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_qvalues[i])

            loss = torch.mean((targets.detach() - qvalues) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.epsilon = max(0, self.epsilon - self.epsilon_decrease)