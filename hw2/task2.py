import torch
from torch import nn
import numpy as np
import gym
import optuna
from matplotlib import animation
import matplotlib.pyplot as plt

import json

import matplotlib.pyplot as plt


class CEM(nn.Module):
    def __init__(self, state_dim, action_n, eps=1):
        super().__init__()
        self.eps = eps
        self.state_dim = state_dim
        self.action_n = action_n

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_n)
        )

        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        result = self.network(_input)
        # result = torch.nn.functional.tanh(result) * torch.tensor(2) работает круто, сходится до -1200, -1100
        result = (torch.sigmoid(result) - torch.tensor(0.5)) * torch.tensor(4)
        return result

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        result = logits.data.numpy()

        # result = logits.data.numpy() + np.random.normal(self.eps)
        # if result < -2:
        #     result = np.array([-2.0])
        # elif result > 2:
        #     result = np.array([2.0])
        return result

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            if len(trajectory['states']) != len(trajectory['actions']):
                trajectory['states'] = trajectory['states'][:-1]
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = self.forward(torch.FloatTensor(np.array(elite_states)))
        elite_actions = torch.LongTensor(np.array(elite_actions))

        mean_ = torch.tensor(1 / len(elite_trajectories))
        loss = mean_ * torch.sqrt(torch.sum(torch.pow(elite_states - elite_actions, 2)))
        result = loss.data.numpy()
        print(f'Loss: {loss}')
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return result


def save_frames_as_gif(frames, path='./', filename='task2_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def get_trajectory(env, agent, trajectory_len, visualize=False, filename=''):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}

    frames = []
    state = env.reset()
    trajectory['states'].append(state)

    for _ in range(trajectory_len):

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward

        if done:
            break

        if visualize:
            frames.append(env.render(mode="rgb_array"))

        trajectory['states'].append(state)

    if visualize:
        save_frames_as_gif(frames)
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [trajectory for trajectory in trajectories if
            trajectory['total_reward'] > quantile]


def run_experiment(episode_n, trajectory_n, trajectory_len, q_param):
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_n = 1

    agent = CEM(state_dim, action_n)
    mean_total_rewards = []
    loss = []

    for episode in range(episode_n):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in
                        range(trajectory_n)]

        mean_total_reward = np.mean(
            [trajectory['total_reward'] for trajectory in trajectories])
        mean_total_rewards.append(mean_total_reward)
        print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
        print(f'Agent eps: {agent.eps}')

        elite_trajectories = get_elite_trajectories(trajectories, q_param)

        if len(elite_trajectories) > 0:
            loss_val = agent.update_policy(elite_trajectories)
            loss.append(loss_val)
            agent.eps -= 1 / episode_n

    get_trajectory(env, agent, trajectory_len, visualize=True)
    return {'scores': mean_total_rewards, 'loss': loss}


def objective(trial):
    episode_n = trial.suggest_int('episode_n', 20, 100)
    trajectory_n = trial.suggest_int('trajectory_n', 100, 1000)
    trajectory_len = trial.suggest_int('trajectory_len', 100, 100_000)
    q_param = trial.suggest_float('q_param', 0.5, 0.8)
    print('episode_n', episode_n)
    print('trajectory_n', trajectory_n)
    print('trajectory_len', trajectory_len)
    print('q_param', q_param)
    result = run_experiment(
        episode_n=episode_n,
        trajectory_n=trajectory_n,
        trajectory_len=trajectory_len,
        q_param=q_param
    )

    return max(result['scores'])


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


if __name__ == '__main__':
    run_experiment(episode_n=48, trajectory_n=372, trajectory_len=40201, q_param=0.623791313581447)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=40, timeout=60*60*5, callbacks=[print_best_callback])


