import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from actor import Actor
from critic import Critic


class Agent():
    def __init__(self, hparams, gamma=0.99):
        self.gamma = gamma
        self.neurons = hparams['neurons']
        self.n_layers = hparams['layers']
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(hparams['lr'])
        self.c_opt = tf.keras.optimizers.Adam(hparams['lr'])
        self.f1 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.f2 = tf.keras.layers.Dense(self.neurons, activation='relu')
        if self.n_layers == 3:
            self.f3 = tf.keras.layers.Dense(self.neurons, activation='relu')

        self.sigma = tf.keras.layers.Dense(1, activation=None)
        self.mu = tf.keras.layers.Dense(1, activation=None)

        self.actor = Actor(hparams)
        self.critic = Critic(hparams)
        self.clip_pram = 0.2

    def act(self, state):
        # TODO może jest lepsza metoda wyboru akcji?
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        x = self.f1(np.array([state]))
        x = self.f2(x)
        if self.n_layers == 3:
            x = self.f3(x)
        sigma = self.sigma(x)
        mu = self.mu(x)
        #dist = tfp.distributions.Uniform(low=-1.0, high=1.0)
        dist = tfp.distributions.Normal(mu, sigma)
        action = dist.sample(sample_shape=(6))
        return action
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []

        for pb, t, op in zip(probability, adv, old_probs):
            t = tf.constant(t)
            op = tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb, op)
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # print(loss)
        return loss

    def learn(self, states, actions, adv, old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p), 6))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
