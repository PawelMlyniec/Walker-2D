import tensorflow as tf


class Actor(tf.keras.Model):
    def __init__(self, hparams):
        super().__init__()
        self.neurons = hparams['neurons']
        self.n_layers = hparams['layers']
        self.f1 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.f2 = tf.keras.layers.Dense(self.neurons, activation='relu')
        if self.n_layers == 3:
            self.f3 = tf.keras.layers.Dense(self.neurons, activation='relu')
        self.a = tf.keras.layers.Dense(6, activation='softmax')

    def call(self, input_data):
        x = self.f1(input_data)
        x = self.f2(x)
        if self.n_layers == 3:
            x = self.f3(x)
        x = self.a(x)
        return x
