import tensorflow as tf
from softAC.actor import Actor
from softAC.critic import Critic
from softAC.value_net import value_net
from softAC.replay_buffer import RBuffer


class Agent:
    def __init__(self, env, hparams):
        self.actor_main = Actor(len(env.action_space.high), hparams)
        self.critic_1 = Critic(hparams)
        self.critic_2 = Critic(hparams)
        self.value_net = value_net(hparams)
        self.target_value_net = value_net(hparams)
        self.batch_size = 64
        self.n_actions = len(env.action_space.high)
        self.a_opt = tf.keras.optimizers.Adam(hparams['lr']/2)
        self.c_opt1 = tf.keras.optimizers.Adam(hparams['lr'])
        self.c_opt2 = tf.keras.optimizers.Adam(hparams['lr'])
        self.v_opt = tf.keras.optimizers.Adam(hparams['lr'])
        self.memory = RBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
        self.trainstep = 0
        self.gamma = 0.99
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.scale = 2

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action, _ = self.actor_main.sample_normal(state)
        return action

    def savexp(self, state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

    def update_target(self):
        self.target_value_net.set_weights(self.value_net.get_weights())

    def train(self):
        if self.memory.cnt < self.batch_size:
            return

        states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3, tf.GradientTape() as tape4:
            value = tf.squeeze(self.value_net(states))
            v_actions, v_log_probs = self.actor_main.sample_normal(states)
            v_q1 = self.critic_1(states, v_actions)
            v_q2 = self.critic_2(states, v_actions)
            v_critic_value = tf.math.minimum(tf.squeeze(v_q1), tf.squeeze(v_q2))
            target_value = v_critic_value - v_log_probs
            value_loss = 0.5 * tf.keras.losses.MSE(target_value, value)

            a_actions, a_log_probs = self.actor_main.sample_normal(states)
            a_q1 = self.critic_1(states, a_actions)
            a_q2 = self.critic_2(states, a_actions)
            a_critic_value = tf.math.minimum(tf.squeeze(a_q1), tf.squeeze(a_q2))
            actor_loss = a_log_probs - a_critic_value
            actor_loss = tf.reduce_mean(actor_loss)

            next_state_value = tf.squeeze(self.target_value_net(next_states))
            q_hat = self.scale * rewards + self.gamma * next_state_value * dones
            c_q1 = self.critic_1(states, actions)
            c_q2 = self.critic_2(states, actions)
            critic_loss1 = 0.5 * tf.keras.losses.MSE(q_hat, tf.squeeze(c_q1))
            critic_loss2 = 0.5 * tf.keras.losses.MSE(q_hat, tf.squeeze(c_q2))

        grads1 = tape1.gradient(value_loss, self.value_net.trainable_variables)
        grads2 = tape2.gradient(actor_loss, self.actor_main.trainable_variables)
        grads3 = tape3.gradient(critic_loss1, self.critic_1.trainable_variables)
        grads4 = tape4.gradient(critic_loss2, self.critic_2.trainable_variables)
        self.v_opt.apply_gradients(zip(grads1, self.value_net.trainable_variables))
        self.a_opt.apply_gradients(zip(grads2, self.actor_main.trainable_variables))
        self.c_opt1.apply_gradients(zip(grads3, self.critic_1.trainable_variables))
        self.c_opt2.apply_gradients(zip(grads4, self.critic_2.trainable_variables))

        self.trainstep += 1
        self.update_target()
