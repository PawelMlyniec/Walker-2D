import tensorflow as tf
import numpy as np
import gym
import pybulletgym
from agent import Agent
from tensorboard.plugins.hparams import api as hp
import time


def run(hparams, run_dir):
    env = gym.make('Walker2DPyBulletEnv-v0')
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

        with tf.device('GPU:0'):
            agent = Agent(env, hparams)
            tf.random.set_seed(17)

            episods = 30
            ep_reward = []
            total_avgr = []
            target = False

            for s in range(episods):
                start = time.time()

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
                        if int(avg_reward) == 20:
                            target = True

                end = time.time()
                tf.summary.scalar('avg_reward', avg_reward, step=s)
                tf.summary.scalar('total_reward', total_reward, step=s)
                tf.summary.scalar('time', end-start, step=s)

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
                run(hparams, 'logs_ddpg/hparam_tuning/' + run_name)
                session_num += 1

