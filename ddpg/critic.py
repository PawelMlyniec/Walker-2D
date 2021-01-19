import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, hparams):
        super(Critic, self).__init__()
        self.neurons = hparams['neurons']
        self.n_layers = hparams['layers']
        self.f = []
        for l in range(self.n_layers):
            self.f.append(tf.keras.layers.Dense(self.neurons, activation='relu'))

        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputstate, action):
        x = tf.concat([inputstate, action], axis=1)
        for f in self.f:
            x = f(x)
        v = self.v(x)
        return v