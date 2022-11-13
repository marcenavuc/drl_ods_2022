import time

import gym
import numpy as np


env = gym.make('Taxi-v3')
STATE_N = env.observation_space.n
ACTION_N = env.action_space.n


class CEM:
    def __init__(self,
                 state_n: int,
                 action_n: int,
                 plambda: float = 0.5,
                 llambda: float = 0.,
                 smoothing_type: str = None):
        self.state_n = state_n
        self.action_n = action_n
        self.plambda = plambda
        self.llambda = llambda
        self.sm_type = smoothing_type
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
                new_policy = np.ones(self.action_n) / self.action_n

            # Laplace smoothing
            elif self.sm_type.lower() == 'laplace':
                up = pre_policy[state] + self.llambda
                down = sum(pre_policy[state]) + self.llambda*self.action_n
                new_policy = up / down
            else:
                new_policy = pre_policy[state] / sum(pre_policy[state])

            # Policy smoothing
            if self.sm_type is None:
                self.policy[state] = new_policy
            elif self.sm_type.lower() == 'policy':
                left = self.plambda * new_policy
                right = (1-self.plambda) * self.policy[state]
                self.policy[state] = left + right
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


def test(agent, trajectory_len):
    state = env.reset()
    for _ in range(trajectory_len):

        action = agent.get_action(state)

        state, reward, done, _ = env.step(action)

        env.render()
        time.sleep(0.5)

        if done:
            break


def main():
    agent = CEM(STATE_N, ACTION_N, plambda=0.5, smoothing_type='policy')
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

    test(agent, trajectory_len)


if __name__ == "__main__":
    main()
