import torch
from torch import nn
import numpy as np


class CEM(nn.Module):
    def __init__(self, model: nn.Module, state_dim: int, action_n: int, q_param: float):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.q_param = q_param

        self.model = model
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.model(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss(self.forward(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def fit(env, agent, episode_n, trajectory_n, trajectory_len):
    total_rewards = []
    for episode in range(episode_n):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in
                        range(trajectory_n)]

        mean_total_reward = np.mean(
            [trajectory['total_reward'] for trajectory in trajectories])
        total_rewards.append(mean_total_reward)

        elite_trajectories = get_elite_trajectories(trajectories, agent.q_param)

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)
    return total_rewards


def get_trajectory(env, agent, trajectory_len):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}
    state = env.reset()
    trajectory['states'].append(state)

    for _ in range(trajectory_len):

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward

        if done:
            break

        trajectory['states'].append(state)
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [trajectory for trajectory in trajectories if
            trajectory['total_reward'] > quantile]