import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import tensorflow as tf

if __name__ == '__main__':
    env = gym.make('Walker2DPyBulletEnv-v0')
    env.render() # call this before env.reset, if you want a window showing the environment
    env.reset()  # should return a state vector if everything worked 