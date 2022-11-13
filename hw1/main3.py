import time

import gym
import numpy as np


env = gym.make('Taxi-v3')
STATE_N = env.observation_space.n
ACTION_N = env.action_space.n
M = 4


class CEM:
    def __init__(self, state_n: int, action_n: int, m: int):
        self.state_n = state_n
        self.action_n = action_n
        self.m = m
        self.policy = np.ones((self.state_n, self.action_n)) / self.action_n
        self.make_policies()

    def make_policies(self):
        actions = np.arange(self.action_n)
        self.policies = []
        for i in range(self.m):
            policy = np.zeros((self.state_n, self.action_n))
            idx = [np.random.choice(actions, p=self.policy[j]) for j in range(self.state_n)]
            policy[np.arange(self.state_n), idx] = idx  # It will help to determine index of action
            self.policies.append(policy)

    def get_action(self, state, policy_id):
        return int(self.policies[policy_id][state].max())

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

        self.make_policies()
        return None


def get_trajectory(agent, policy_id, trajectory_len):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}

    state = env.reset()
    trajectory['states'].append(state)

    for _ in range(trajectory_len):

        action = agent.get_action(state, policy_id)
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
    agent = CEM(STATE_N, ACTION_N, M)
    episode_n = 100
    trajectory_n = 250
    trajectory_len = 20
    q_param = 0.35

    for i in range(episode_n):
        trajectories = []

        for m in range(agent.m):
            for _ in range(trajectory_n):
                trajectories.append(get_trajectory(agent, m, trajectory_len))

        mean_total_reward = np.mean([t['total_reward'] for t in trajectories])
        print("Mean total reward", mean_total_reward)

        elite_trajectories = get_elite_trajectories(trajectories, q_param)
        print("Amount of elite", len(elite_trajectories))
        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)

    test(agent, trajectory_len)


if __name__ == "__main__":
    main()
