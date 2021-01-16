import numpy as np
import tensorflow as tf
import gym
import pybulletgym
from utils import test_reward, preprocess1
from agent import Agent
from tensorboard.plugins.hparams import api as hp


def run(hparams, run_dir):
    env = gym.make("Walker2DPyBulletEnv-v0")
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

        with tf.device('GPU:0'):
            tf.random.set_seed(336699)
            agent = Agent(hparams)
            episodes = 5000
            ep_reward = []
            total_avgr = []
            target = False
            best_reward = 0
            avg_rewards_list = []

            for s in range(episodes):
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

                while not done:
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
                    # agent.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
                    # agent.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
                    best_reward = avg_reward
                if best_reward == 200:
                    target = True

    env.close()

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    HP_NEURONS = hp.HParam('neurons', hp.Discrete([128, 256]))
    HP_LAYERS = hp.HParam('layers', hp.Discrete([2, 3]))
    HP_LR = hp.HParam('lr', hp.Discrete([0.001, 0.0005]))
    session_num = 0

    for neuron in HP_NEURONS.domain.values:
        for layer in HP_LAYERS.domain.values:
            for lr in HP_LR.domain.values:
                hparams = {
                    'neurons': neuron,
                    'layers': layer,
                    'lr': lr,
                }
                run_name = str({h: hparams[h] for h in hparams})
                print('--- Starting trial: %s' % run_name)
                print({h: hparams[h] for h in hparams})
                run(hparams, 'logs/hparam_tuning/' + run_name)
                session_num += 1
