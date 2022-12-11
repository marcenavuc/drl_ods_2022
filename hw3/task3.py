from frozen_lake import FrozenLakeEnv

import pandas as pd
import numpy as np
from tqdm import tqdm

env = FrozenLakeEnv()


def init_policy():
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def policy_evaluation_step(policy, values, gamma):
    q_values = get_q_values(values, gamma)

    new_values = {}
    for state in env.get_all_states():
        new_values[state] = 0
        for action in env.get_possible_actions(state):
            new_values[state] += policy[state][action] * q_values[state][action]

    return new_values


def init_values():
    return {state: 0 for state in env.get_all_states()}


def get_q_values(values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                reward = env.get_reward(state, action, next_state)
                transition_prob = env.get_transition_prob(state, action,
                                                          next_state)
                next_value = values[next_state]
                q_values[state][action] += transition_prob * (
                        reward + gamma * next_value)
    return q_values


def policy_evaluation(policy, gamma, evaluation_step_n):
    values = init_values()
    for _ in range(evaluation_step_n):
        values = policy_evaluation_step(policy, values, gamma)
    q_values = get_q_values(values, gamma)
    return q_values


def policy_evaluation_mod(q_values, policy, gamma, evaluation_step_n):
    values = init_values()
    for state in values:
        values[state] = max(q_values[state].values()) if len(q_values[state]) > 0 else 0
    for _ in range(evaluation_step_n):
        values = policy_evaluation_step(policy, values, gamma)
    return get_q_values(values, gamma)


def policy_improvement(q_values):
    new_policy = {}
    for state in env.get_all_states():
        new_policy[state] = {}
        max_action = None
        max_q_value = float('-inf')
        for action in env.get_possible_actions(state):
            if q_values[state][action] > max_q_value:
                max_q_value = q_values[state][action]
                max_action = action
        for action in env.get_possible_actions(state):
            new_policy[state][action] = 1 if action == max_action else 0
    return new_policy


def test_policy(policy, render=False):
    total_reward = 0
    state = env.reset()
    for _ in range(100):
        action = np.random.choice(env.get_possible_actions(state),
                                  p=list(policy[state].values()))
        state, reward, done, _ = env.step(action)
        if render:
            env.render()

        total_reward += reward

        if done or env.is_terminal(state):
            break
    return total_reward


def run_experiment(iteration_n,
                   evaluation_step_n,
                   gamma,
                   render=False,
                   kind='default'):
    policy = init_policy()
    for i in range(iteration_n):
        if kind == 'default' or i == 0:
            q_values = policy_evaluation(policy, gamma, evaluation_step_n)
        elif kind == 'mod':
            q_values = policy_evaluation_mod(q_values, policy, gamma, evaluation_step_n)
        policy = policy_improvement(q_values)

    total_rewards = []
    for _ in range(1000):
        total_rewards.append(test_policy(policy, render))
    return np.mean(total_rewards)


def main():
    seed = 1
    n_iterations = 50
    np.random.seed(seed)
    result_default = []
    result_mod = []
    gamma_space = np.linspace(0.01, 0.99, 100)
    iteration_space = [int(_) for _ in np.linspace(10, 100, 20)]
    evaluation_space = [int(_) for _ in np.linspace(10, 100, 20)]
    res = {'g': [], 'i': [], 'e': []}

    for _ in tqdm(range(n_iterations)):
        g = np.random.choice(gamma_space)
        i = np.random.choice(iteration_space)
        e = np.random.choice(evaluation_space)
        res['g'].append(g)
        res['i'].append(i)
        res['e'].append(e)

        result_default.append(run_experiment(i, e, g, kind='default'))
        result_mod.append(run_experiment(i, e, g, kind='mod'))

    result_df = pd.DataFrame({
        'gamma': res['g'],
        'iteration_n': res['i'],
        'evaluation_step_n': res['e'],
        'mean_total_reward_default': result_default,
        'mean_total_reward_mod': result_mod
    })

    result_df.to_csv('task2.csv')
    print(
        result_df.sort_values('mean_total_reward_mod', ascending=False).head()
    )


if __name__ == '__main__':
    main()
