import numpy as np

from keras.layers import Dense

from keras.optimizers import Adam
from keras.models import Sequential
import os


class A2C:
    def __init__(self, state_size, action_size, load_model=False):
        self.render = True
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        self.accumulated_steps = []
        self.max_steps = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        if load_model:
            self.actor.load_weights("agents/save_model/a2c_discrete_actor.h5")
            self.critic.load_weights("agents/save_model/a2c_discrete_critic.h5")
        else:
            self.actor = self.build_actor()
            self.critic = self.build_critic()

    def save_model(self):
        self.actor.save_weights("agents/save_model/a2c_discrete_actor.h5")
        self.critic.save_weights("agents/save_model/a2c_discrete_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(50, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, p=policy)

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        self.accumulated_steps.append((state, action, reward))

        # only update model after max_steps
        if len(self.accumulated_steps) <= self.max_steps and not done:
            return[0]

        states = [step[0] for step in self.accumulated_steps]
        actions = [step[1] for step in self.accumulated_steps]
        rewards = [step[2] for step in self.accumulated_steps]

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

        actor_target = np.zeros((len(self.accumulated_steps)-1, self.action_size))
        for i, x in enumerate(actions[:-1]):
            actor_target[i][x] = advantages[i]

        states = np.array([state[0] for state in states[:-1]])
        actor_target = np.array(actor_target)
        critic_target = np.array(critic_target)
        self.critic.fit(states, critic_target, epochs=1, verbose=0)
        self.actor.fit(states, actor_target, epochs=1, verbose=0)

        self.accumulated_steps = [self.accumulated_steps[-1]]