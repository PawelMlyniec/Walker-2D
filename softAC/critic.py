import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, hparams):
        super(Critic, self).__init__()
        self.neurons = hparams['neurons']
        self.n_layers = hparams['layers']
        self.f1 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.f2 = tf.keras.layers.Dense(self.neurons, activation='relu')
        if self.n_layers == 3:
            self.f3 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputstate, action):
        x = self.f1(tf.concat([inputstate, action], axis=1))
        x = self.f2(x)
        if self.n_layers == 3:
            x = self.f3(x)
        x = self.q(x)
        return x
