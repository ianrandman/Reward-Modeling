import numpy as np
import keras.backend as K

from keras.layers import Input, Dense, Lambda

from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model, save_model
import math
import os

import tensorflow as tf
import json

K.clear_session()


class A2C_Continuous:
    def __init__(self, state_size, action_size, action_high, env, load_model=False):
        self.save_dir = os.path.dirname(os.path.abspath(__file__ + '/../')) + '/save_model/' + env + '/'
        
        self.render = True
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.actor_loss_history = []
        self.critic_loss_history = []

        self.accumulated_steps = []
        self.max_steps = 25

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.8
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        if load_model:
            with open(self.save_dir + "/actor_loss_history.json", 'r') as f:
                self.actor_loss_history = json.load(f)
            with open(self.save_dir + "/critic_loss_history.json", 'r') as f:
                self.critic_loss_history = json.load(f)
            self.actor.load_weights(self.save_dir+"/a2c_continuous_actor.h5")
            self.critic.load_weights(self.save_dir+"/a2c_continuous_critic.h5")

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_dir + "/actor_loss_history.json", "w+") as f:
            json.dump(self.actor_loss_history, f)
        with open(self.save_dir + "/critic_loss_history.json", "w+") as f:
            json.dump(self.critic_loss_history, f)
        self.actor.save_weights(self.save_dir+"/a2c_continuous_actor.h5")
        self.critic.save_weights(self.save_dir+"/a2c_continuous_critic.h5")

    def tensor_pdf(self, mu, sigma, x):
        # math.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * (2 * math.pi ** 2)**0.5)
        exp_first = tf.math.multiply(tf.constant(-0.5), tf.math.square(x - mu))
        top = tf.math.exp(tf.math.divide(exp_first, tf.math.square(sigma)))
        bottom = tf.math.multiply(tf.constant((2 * math.pi) ** 0.5), sigma)
        return tf.math.divide(top, bottom)

    def actor_loss_wrapper(self, mu, sigma):
        def actor_loss(advantage, predicted_action):
            pdf = self.tensor_pdf(mu, sigma, predicted_action)
            return -1 * tf.math.log(pdf) * advantage
        return actor_loss

    def sample_dist(self, mu_sigma):
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
        dist = tf.contrib.distributions.Normal(mu, sigma)
        return tf.clip_by_value(dist.sample(1), -self.action_high, self.action_high)

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        input = Input(shape=(self.state_size,))
        hidden = Dense(50, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l1')(input)
        mu = Dense(self.action_size, name='mu', activation='tanh', kernel_initializer='he_uniform')(hidden)
        sigma = Dense(self.action_size, name='sigma', activation='softplus', kernel_initializer='he_uniform')(hidden)
        actions = Lambda(self.sample_dist)([mu, sigma])
        actor = Model(inputs=input, outputs=actions)
        actor._make_predict_function()

        actor.summary()
        actor.compile(loss=self.actor_loss_wrapper(mu, sigma),
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(50, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l1'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))

        critic._make_predict_function()

        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    def get_action(self, state):
        return self.actor.predict(np.reshape(state, [1, self.state_size]))[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        self.accumulated_steps.append((state, reward))

        # only update model after max_steps
        if len(self.accumulated_steps) <= self.max_steps and not done:
            return[0]

        states = [step[0] for step in self.accumulated_steps]
        rewards = [step[1] for step in self.accumulated_steps]

        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        eps = np.finfo(np.float32).eps.item()
        rewards = [(reward - rewards_mean) / (rewards_std + eps) for reward in rewards]

        v_hats = [self.critic.predict(state)[0] for state in states]
        v_actuals = []

        # calculate discounted actual rewards plus the discounted final v_hat
        discounted_rewards = [x*(np.power(self.discount_factor, i)) for i, x in enumerate(rewards[:-1])]
        v_actuals.append(sum(discounted_rewards) +
                         v_hats[-1] * np.power(self.discount_factor, len(self.accumulated_steps)-1))

        # update other v_actuals using previous v_actual minus reward
        for i in range(1, len(self.accumulated_steps)-1):
            v_actuals.append(v_actuals[i-1] - discounted_rewards[i-1])

        advantages = np.subtract(v_actuals, v_hats[:-1])
        critic_target = v_actuals

        states = np.array([state[0] for state in states[:-1]])
        actor_target = np.array(advantages)
        critic_target = np.array(critic_target)
        critic_history = self.critic.fit(states, critic_target, epochs=1, verbose=0)
        actor_history = self.actor.fit(states, actor_target, epochs=1, verbose=0)
        self.critic_loss_history.extend(critic_history.history["loss"])
        self.actor_loss_history.extend(actor_history.history["loss"])

        self.accumulated_steps = [self.accumulated_steps[-1]]
