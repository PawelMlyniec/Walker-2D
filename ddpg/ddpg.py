import tensorflow as tf
import numpy as np
import gym
import pybulletgym
from agent import Agent

if __name__ == '__main__':
    env = gym.make('Walker2DPyBulletEnv-v0')
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    action_low = env.action_space.low
    action_high = env.action_space.high
    print(state_low)
    print(state_high)
    print(action_low)
    print(action_high)

    agent = Agent()
    tf.random.set_seed(17)

    episods = 2000
    ep_reward = []
    total_avgr = []
    target = False

    for s in range(episods):
        if target:
            break
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.savexp(state, next_state, action, done, reward)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))
                if int(avg_reward) == 200:
                    target = True

