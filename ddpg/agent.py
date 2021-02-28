import tensorflow as tf
from ddpg.actor import Actor
from ddpg.critic import Critic
from ddpg.replay_buffer import RBuffer


class Agent():
    def __init__(self, env, hparams):
        n_action = len(env.action_space.high)
        self.actor_main = Actor(n_action, hparams)
        self.actor_target = Actor(n_action, hparams)
        self.critic_main = Critic(hparams)
        self.critic_target = Critic(hparams)
        self.batch_size = 64
        self.n_actions = len(env.action_space.high)
        self.a_opt = tf.keras.optimizers.Adam(hparams['lr'])
        # self.actor_target = tf.keras.optimizers.Adam(.001)
        self.c_opt = tf.keras.optimizers.Adam(hparams['lr'])
        # self.critic_target = tf.keras.optimizers.Adam(.002)
        self.memory = RBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
        self.trainstep = 0
        self.replace = 5
        self.gamma = 0.99
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]

    def act(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor_main(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)

        actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
        # print(actions)
        return actions[0]

    def savexp(self, state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

    def update_target(self):
        self.actor_target.set_weights(self.actor_main.get_weights())
        self.critic_target.set_weights(self.critic_main.get_weights())

    def train(self):
        if self.memory.cnt < self.batch_size:
            return

        states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        # dones = tf.convert_to_tensor(dones, dtype= tf.bool)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            target_actions = self.actor_target(next_states)
            target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic_main(states, actions), 1)
            target_values = rewards + self.gamma * target_next_state_values * dones
            critic_loss = tf.keras.losses.MSE(target_values, critic_value)

            new_policy_actions = self.actor_main(states)
            actor_loss = -self.critic_main(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        grads1 = tape1.gradient(actor_loss, self.actor_main.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic_main.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor_main.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic_main.trainable_variables))

        if self.trainstep % self.replace == 0:
            self.update_target()

        self.trainstep += 1
