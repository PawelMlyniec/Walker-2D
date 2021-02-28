import dill as dill
import tensorflow as tf
import numpy as np
import gym
import pybulletgym
from ddpg.agent import Agent
from tensorboard.plugins.hparams import api as hp
import time


def run(hparams, run_dir, best_reward, fps):
    env = gym.make('Walker2DPyBulletEnv-v0')
    env.metadata['video.frames_per_second'] = fps  # base fps is 60
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

        with tf.device('GPU:0'):
            agent = Agent(env, hparams)
            tf.random.set_seed(17)

            episods = 1000
            ep_reward = []
            total_avgr = []
            total_time = 0
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
                        if int(avg_reward) >= 550:
                            target = True
                        # if avg_reward > best_reward:
                        #     print('best reward=' + str(avg_reward))
                        #     best_reward = avg_reward
                        #     with open("./models/ddpgAgent20", "wb") as dill_file:
                        #         dill.dump(agent, dill_file)

                end = time.time()
                tf.summary.scalar('avg_reward', avg_reward, step=s)
                tf.summary.scalar('total_reward', total_reward, step=s)
                total_time += end-start
                tf.summary.scalar('time', total_time, step=s)
                if s%40 == 0:
                    with open(r"models\agent_ddpg_1000_550_120fps", "wb+") as dill_file:
                        dill.dump(agent, dill_file)

    env.close()
    return best_reward


def walk():
    with open(r"models\agent_ddpg3000_500", "rb") as dill_file:
        agent = dill.load(dill_file)
    env = gym.make('Walker2DPyBulletEnv-v0')
    tf.random.set_seed(336699)

    ep_reward = []
    total_avgr = []
    total_time = 0

    total_reward = 0

    env.render()
    state = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.savexp(state, next_state, action, done, reward)
        state = next_state
        total_reward += reward
        if done:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} steps is {} and avg reward is {}".format(1, total_reward, avg_reward))
    env.close()


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    HP_NEURONS = hp.HParam('neurons', hp.Discrete([128]))
    HP_LAYERS = hp.HParam('layers', hp.Discrete([3]))
    HP_LR = hp.HParam('lr', hp.Discrete([0.001]))
    fps = 60
    session_num = 0
    best_reward = 0
    for neuron in HP_NEURONS.domain.values:
        break
        for layer in HP_LAYERS.domain.values:
            for lr in HP_LR.domain.values:
                hparams = {
                    'neurons': neuron,
                    'layers': layer,
                    'lr': lr,
                }
                #run_name = str({h: hparams[h] for h in hparams})
                run_name = str(f'{list(hparams.items())}, fps={fps}')
                print('--- Starting trial: %s' % run_name)
                print({h: hparams[h] for h in hparams})
                reward = run(hparams, 'logs_ddpg\\hparam_tuning2\\' + run_name, best_reward, fps)
                session_num += 1
    walk()
