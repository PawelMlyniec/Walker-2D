import tensorflow as tf
from copy import deepcopy

class Actor(tf.keras.Model):
    def __init__(self, no_action, hparams):
        super(Actor, self).__init__()
        self.neurons = hparams['neurons']
        self.n_layers = hparams['layers']
        self.f = []
        for l in range(self.n_layers):
            self.f.append(tf.keras.layers.Dense(self.neurons, activation='relu'))
        self.mu = tf.keras.layers.Dense(no_action, activation='tanh')

    def call(self, state):
        x = deepcopy(state)
        for f in self.f:
            x = f(x)
        x = self.mu(x)
        return x