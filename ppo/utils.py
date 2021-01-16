import numpy as np
import tensorflow as tf


def test_reward(env, agent):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        actor = agent.actor(np.array([state]))
        action = tf.math.argmax(actor)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward


def preprocess1(states, actions, rewards, done, values, gamma, dones):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
        g = delta + gamma * lmbda * dones[i] * g
        returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv
