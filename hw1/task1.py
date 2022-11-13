import os

import gym
import pandas as pd
import numpy as np
import plotly.graph_objects as go

SEED = os.getenv("SEED", 1)
np.random.seed(SEED)
env = gym.make('Taxi-v3')
STATE_N = env.observation_space.n
ACTION_N = env.action_space.n


class CEM:
    def __init__(self, state_n: int, action_n: int, q_param: float):
        self.state_n = state_n
        self.action_n = action_n
        self.q_param = q_param
        self.policy = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        return int(np.random.choice(np.arange(self.action_n), p=self.policy[state]))

    def update_policy(self, elite_trajectories):
        pre_policy = np.zeros((self.state_n, self.action_n))

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                pre_policy[state][action] += 1

        for state in range(self.state_n):
            if sum(pre_policy[state]) == 0:
                self.policy[state] = np.ones(self.action_n) / self.action_n
            else:
                self.policy[state] = pre_policy[state] / sum(pre_policy[state])

        return None

    def get_trajectory(self, trajectory_len):
        trajectory = {'states': [], 'actions': [], 'total_reward': 0}

        state = env.reset()
        trajectory['states'].append(state)

        for _ in range(trajectory_len):

            action = self.get_action(state)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            trajectory['total_reward'] += reward

            if done:
                break

            trajectory['states'].append(state)

        return trajectory

    def get_elite_trajectories(self, trajectories):
        total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
        quantile = np.quantile(total_rewards, q=self.q_param)
        return [trajectory for trajectory in trajectories if
                trajectory['total_reward'] > quantile]

    def fit(self, episode_n, trajectory_n, trajectory_len):
        total_rewards = []

        for i in range(episode_n):
            trajectories = [self.get_trajectory(trajectory_len) for _ in range(trajectory_n)]

            mean_total_reward = np.mean(
                [trajectory['total_reward'] for trajectory in trajectories])
            total_rewards.append(mean_total_reward)

            elite_trajectories = self.get_elite_trajectories(trajectories)

            if len(elite_trajectories) > 0:
                self.update_policy(elite_trajectories)
        return total_rewards
    
    def get_scores(self, trajectory_len, trajectory_n):
        trajectories = [self.get_trajectory(trajectory_len) for _ in range(trajectory_n)]
        return np.mean([trajectory['total_reward'] for trajectory in trajectories])
        

def run_example_experiment():
    agent = CEM(STATE_N, ACTION_N, q_param=0.5)
    result = agent.fit(episode_n=100,
              trajectory_n=250,
              trajectory_len=10000)
    print("Score", agent.get_scores(trajectory_len=10000, trajectory_n=250))
    df = pd.DataFrame({'steps': np.arange(100),
                       'mean_total_reward': result, "best": 0})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['steps'], y=df['mean_total_reward'],
                             name='total_reward'))
    fig.add_trace(go.Scatter(x=df['steps'], y=df['best'], name='0'))
    fig.write_image("task1_example.png")


def find_best_score():
    N_EXPERIMENTS = 30

    params = {
        "q_param": [0.3, 0.4, 0.5, 0.6, 0.7],
        "episode_n": [12, 25, 50],
        "trajectory_n": [250, 500],
        "trajectory_len": [500, 1000, 2000, 10 ** 4]
    }

    experiments = []
    experiments_val = []
    for i in range(N_EXPERIMENTS):
        exp = {}
        for key in params:
            exp[key] = np.random.choice(params[key])
        agent = CEM(STATE_N, ACTION_N, q_param=exp['q_param'])
        result = agent.fit(episode_n=exp['episode_n'],
                           trajectory_n=exp['trajectory_n'],
                           trajectory_len=exp['trajectory_len'])
        exp['result'] = agent.get_scores(trajectory_len=exp['trajectory_len'],
                                         trajectory_n=exp['trajectory_n'])
        experiments.append(exp)
        print(i, exp)
        experiments_val.append(result)

    exp_df = pd.DataFrame(experiments)
    print("best result", exp_df[exp_df['result'] == exp_df['result'].max()])

    best_iter = exp_df[exp_df['result'] == exp_df['result'].max()].index[0]
    steps = np.arange(exp_df.loc[best_iter]['episode_n'])
    mean_total_reward = experiments_val[best_iter]

    df = pd.DataFrame(
        {'steps': steps, 'mean_total_reward': mean_total_reward, "best": 0})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['steps'], y=df['mean_total_reward'],
                             name='total_reward'))
    fig.add_trace(go.Scatter(x=df['steps'], y=df['best'], name='0'))
    fig.write_image("task1_best.png")


def main():
    run_example_experiment()
    find_best_score()


if __name__ == "__main__":
    main()
