from frozen_lake import FrozenLakeEnv
import numpy as np
import time

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


iteration_n = 20
evaluation_step_n = 20
gamma = 0.9

policy = init_policy()
for _ in range(iteration_n):
    q_values = policy_evaluation(policy, gamma, evaluation_step_n)
    policy = policy_improvement(q_values)

total_reward = 0
state = env.reset()
for _ in range(100):
    action = np.random.choice(env.get_possible_actions(state),
                              p=list(policy[state].values()))
    state, reward, done, _ = env.step(action)

    time.sleep(1)
    env.render()

    total_reward += reward

    if done or env.is_terminal(state):
        break

print(f'total reward = {total_reward}')
