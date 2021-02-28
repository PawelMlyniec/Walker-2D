import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import tensorflow as tf
from softAC.main_sac import run as sac_run
from ddpg.ddpg import run as ddpg_run
from tensorboard.plugins.hparams import api as hp

if __name__ == '__main__':
    HP_NEURONS = hp.HParam('neurons', hp.Discrete([128]))
    HP_LAYERS = hp.HParam('layers', hp.Discrete([2]))
    HP_LR = hp.HParam('lr', hp.Discrete([0.001]))
    HP_FPS = [120]
    session_num = 0
    best_reward = 0

    for neuron in HP_NEURONS.domain.values:
        for layer in HP_LAYERS.domain.values:
            for lr in HP_LR.domain.values:
                for fps in HP_FPS:
                    hparams = {
                        'neurons': neuron,
                        'layers': layer,
                        'lr': lr,
                    }
                    # run_name = str({h: hparams[h] for h in hparams})
                    run_name = str(f'{list(hparams.items())}, fps={fps}')
                    print('--- Starting trial: %s' % run_name)
                    print({h: hparams[h] for h in hparams})
                    sac_run(hparams, 'logs_sac\\hparam_tuning\\' + run_name, fps)
                    #ddpg_run(hparams, 'logs_ddpg\\hparam_tuning\\' + run_name, best_reward, fps)
                    session_num += 1

    HP_NEURONS = hp.HParam('neurons', hp.Discrete([128]))
    HP_LAYERS = hp.HParam('layers', hp.Discrete([3]))
    HP_LR = hp.HParam('lr', hp.Discrete([0.001]))

    for neuron in HP_NEURONS.domain.values:
        for layer in HP_LAYERS.domain.values:
            for lr in HP_LR.domain.values:
                for fps in HP_FPS:
                    hparams = {
                        'neurons': neuron,
                        'layers': layer,
                        'lr': lr,
                    }
                    # run_name = str({h: hparams[h] for h in hparams})
                    run_name = str(f'{list(hparams.items())}, fps={fps}')
                    print('--- Starting trial: %s' % run_name)
                    print({h: hparams[h] for h in hparams})
                    #sac_run(hparams, 'logs_sac\\hparam_tuning\\' + run_name, fps)
                    ddpg_run(hparams, 'logs_ddpg\\hparam_tuning\\' + run_name, best_reward, fps)
                    session_num += 1
