import gym
import torch
import torch.nn as nn
import numpy as np
import random

class NN(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, action_n)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        hidden = self.linear1(state)
        hidden = self.relu(hidden)
        hidden = self.linear2(hidden)
        hidden = self.relu(hidden)
        qvalues = self.linear3(hidden)
        return qvalues

    
class DQN():
    def __init__(self, action_n, model, batch_size, gamma, lr, trajectory_n):
        self.action_n = action_n
        self.model = model
        self.epsilon = 1
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decrease = 1 / trajectory_n
        self.memory = []
        self.optimazer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
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
            self.optimazer.step()
            self.optimazer.zero_grad()
            
            self.epsilon = max(0, self.epsilon - self.epsilon_decrease)

    
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n

trajectory_n = 50
trajectory_len = 500
batch_size = 64
gamma = 0.99
lr = 1e-2

model = NN(state_dim, action_n)
agent = DQN(action_n, model, batch_size, gamma, lr, trajectory_n)

for trajectory_i in range(trajectory_n):
    total_rewards = 0
    
    state = env.reset()
    for i in range(trajectory_len):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        total_rewards += reward

        agent.fit(state, action, reward, done, next_state)

        state = next_state

        env.render()

        if done:
            break
            
    print(f'trajectory {trajectory_i}, total_rewards = {total_rewards}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



