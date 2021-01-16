import tensorflow as tf
import tensorflow_probability as tfp


class Actor(tf.keras.Model):
    def __init__(self, no_action, hparams):
        super(Actor, self).__init__()
        self.neurons = hparams['neurons']
        self.n_layers = hparams['layers']
        self.f1 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.f2 = tf.keras.layers.Dense(self.neurons, activation='relu')
        if self.n_layers == 3:
            self.f3 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.mu = tf.keras.layers.Dense(no_action, activation=None)
        self.sigma = tf.keras.layers.Dense(no_action, activation=None)
        self.min_action = -1
        self.max_action = 1
        self.repram = 1e-6

    def call(self, state):
        x = self.f1(state)
        x = self.f2(x)
        if self.n_layers == 3:
            x = self.f3(x)
        mu = self.mu(x)
        s = self.sigma(x)
        s = tf.clip_by_value(s, self.repram, 1)
        return mu, s

    def sample_normal(self, state):
        mu, sigma = self(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        actions = probabilities.sample()

        action = tf.math.scalar_mul(tf.constant(self.max_action, dtype=tf.float32), tf.math.tanh(actions))
        action = tf.squeeze(action)
        log_prob = probabilities.log_prob(actions)
        log_prob -= tf.math.log(1 - tf.math.pow(action, 2) + self.repram)
        log_prob = tf.reduce_sum(log_prob, axis=1)

        return action, log_prob
