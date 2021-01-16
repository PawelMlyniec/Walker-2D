import numpy as np
import tensorflow as tf
import gym
import pybulletgym
from utils import test_reward, preprocess1
from agent import Agent


if __name__ == '__main__':
    env = gym.make('Walker2DPyBulletEnv-v0')
    tf.random.set_seed(17)
    agent = Agent()
    steps = 5000
    ep_reward = []
    total_avgr = []
    target = False
    best_reward = 0
    avg_rewards_list = []

    for s in range(steps):
        if target:
            break

        done = False
        state = env.reset()
        all_aloss = []
        all_closs = []
        rewards = []
        states = []
        actions = []
        probs = []
        dones = []
        values = []
        print("new episod")

        for e in range(128):

            action = agent.act(state)
            value = agent.critic(np.array([state])).numpy()
            next_state, reward, done, _ = env.step(action)
            dones.append(1 - done)
            rewards.append(reward)
            states.append(state)
            # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            prob = agent.actor(np.array([state]))
            probs.append(prob[0])
            values.append(value[0][0])
            state = next_state
            if done:
                env.reset()

        value = agent.critic(np.array([state])).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs), 6))
        probs = np.stack(probs, axis=0)

        states, actions, returns, adv = preprocess1(states, actions, rewards, dones, values, 1, dones)

        for epocs in range(10):
            al, cl = agent.learn(states, actions, adv, probs, returns)
            # print(f"al{al}")
            # print(f"cl{cl}")

        avg_reward = np.mean([test_reward(env, agent) for _ in range(5)])
        print(f"total test reward is {avg_reward}")
        avg_rewards_list.append(avg_reward)
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            #agent.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            #agent.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            best_reward = avg_reward
        if best_reward == 200:
            target = True
        env.reset()

    env.close()
