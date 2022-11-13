import os

import gym
import numpy as np

SEED = os.getenv("SEED", 1)
env = gym.make('Taxi-v3')
STATE_N = env.observation_space.n
ACTION_N = env.action_space.n


class CEM:
    def __init__(self, state_n: int, action_n: int):
        self.state_n = state_n
        self.action_n = action_n
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


def get_trajectory(agent, trajectory_len):
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


def main():
    agent = CEM(STATE_N, ACTION_N)
    episode_n = 100
    trajectory_n = 250
    trajectory_len = 10**4
    q_param = 0.5

    for i in range(episode_n):
        trajectories = [get_trajectory(agent, trajectory_len) for _ in
                        range(trajectory_n)]

        mean_total_reward = np.mean(
            [trajectory['total_reward'] for trajectory in trajectories])
        if i % 10 == 0:
            print(mean_total_reward)

        elite_trajectories = get_elite_trajectories(trajectories, q_param)

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)


if __name__ == "__main__":
    main()
