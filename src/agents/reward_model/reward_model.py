import numpy as np
import keras.backend as K

from keras.layers import Input, Dense, concatenate, Lambda, Reshape

from keras.optimizers import Adam
from keras.models import Model

import tensorflow as tf
import os


class Ensemble:
    def __init__(self, state_size, action_size, num_steps, env, num_agents=3, load_model=False):
        model = RewardModel(state_size, action_size, num_steps, env, load_model)
        self.model = model
        self.model_list = num_agents * [model.training_model]

    def get_variance(self, seg1_dirty, seg2_dirty):
        seg_1s = [[x['state'], x['actions']] for x in seg1_dirty]
        seg_2s = [[x['state'], x['actions']] for x in seg2_dirty]

        act1 = []
        obs1 = []
        act2 = []
        obs2 = []
        for step1, step2 in zip(seg_1s, seg_2s):
            act1.append(step1[1])
            act2.append(step2[1])
            obs1.append(step1[0])
            obs2.append(step2[0])
        inputs = [[[obs1, obs2]], [[act1, act2]]]


        prefer_1 = []
        prefer_2 = []
        for model in self.model_list:
            p_1_over_2, p_2_over_1 = model.predict(inputs)
            prefer_1.append(p_1_over_2)
            prefer_2.append(p_2_over_1)
        return np.var(np.array(prefer_1)) + np.var(np.array(prefer_2))


class RewardModel:
    def __init__(self, state_size, action_size, num_steps, env, load_model):
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.num_steps = num_steps

        # hyperparameters
        self.lr = 0.001

        # create model
        if load_model:
            self.training_model.load_weights("agents/save_model/"+env+"/reward_model_training.h5")
            self.model.load_weights("agents/save_model/"+env+"/reward_model_mlp.h5")
        else:
            self.training_model, self.model = self.build_model()

    def save_model(self, env):
        if not os.path.exists("agents/save_model/"+env):
            os.makedirs("agents/save_model/"+env)
        self.training_model.save_weights("agents/save_model/"+env+"/reward_model_training.h5")
        self.model.save_weights("agents/save_model/"+env+"/reward_model_mlp.h5")

    def probability_lambda(self, seg_rewards):
        seg_1_rewards = tf.gather(seg_rewards, 0)
        seg_2_rewards = tf.gather(seg_rewards, 1)
        norm_seg_1 = tf.exp(tf.reduce_sum(seg_1_rewards))
        norm_seg_2 = tf.exp(tf.reduce_sum(seg_2_rewards))
        return tf.divide(norm_seg_1, tf.add(norm_seg_1, norm_seg_2))

    def probability_lambda_reverse(self, seg_rewards):
        seg_1_rewards = tf.gather(seg_rewards, 0)
        seg_2_rewards = tf.gather(seg_rewards, 1)
        return self.probability_lambda(tf.concat([seg_2_rewards, seg_1_rewards], axis=0))

    def cust_loss_wrapper(self, p_1_over_2, p_2_over_1):
        def cust_loss(pref, predicted):
            mu_1, mu_2 = pref, 1 - pref
            return -1 * K.sum(tf.add(tf.multiply(mu_1, tf.log(p_1_over_2)), tf.multiply(mu_2, tf.log(p_2_over_1))))
        return cust_loss

    def build_model(self):
        states_input = Input(shape=(2, self.num_steps, self.state_size,))
        states_reshape = Reshape([2 * self.num_steps, self.state_size])(states_input)
        actions_input = Input(shape=(2, self.num_steps, self.action_size,))
        actions_reshape = Reshape([2 * self.num_steps, self.action_size])(actions_input)

        # MLP
        ####################################################################################

        # state branch
        x = Dense(64, activation='relu', kernel_regularizer='l1', name='state_mlp0')(states_reshape)
        x = Dense(32, activation='relu', name='state_mlp1')(x)
        x = Dense(16, activation='relu', name='state_mlp2')(x)

        # action branch
        y = Dense(16, activation='relu', name='actions_mlp0')(actions_reshape)
        y = Dense(32, activation='relu', kernel_regularizer='l1', name='actions_mlp1')(y)
        y = Dense(16, activation='relu', name='actions_mlp2')(y)

        # combine branches
        combined = concatenate([x, y], name='concat')

        # learn after combination
        z = Dense(8, activation='relu', name='mlp_output0')(combined)
        z = Dense(1, activation='tanh', name='mlp_output1')(z)

        ####################################################################################

        seg_rewards = Reshape([2, self.num_steps])(z)

        p_1_over_2 = Lambda(self.probability_lambda, name='first_lambda')(seg_rewards)
        p_2_over_1 = Lambda(self.probability_lambda_reverse, name='second_lambda')(seg_rewards)

        training_model = Model(inputs=[states_input, actions_input], outputs=[p_1_over_2, p_2_over_1])

        training_model.summary()
        training_model.compile(loss=self.cust_loss_wrapper(p_1_over_2, p_2_over_1), optimizer=Adam(lr=self.lr))

        # create MLP model
        states_inp = Input(shape=(self.state_size,))
        actions_inp = Input(shape=(self.action_size,))

        state_model = training_model.get_layer('state_mlp0')(states_inp)
        state_model = training_model.get_layer('state_mlp1')(state_model)
        state_model = training_model.get_layer('state_mlp2')(state_model)

        actions_model = training_model.get_layer('actions_mlp0')(actions_inp)
        actions_model = training_model.get_layer('actions_mlp1')(actions_model)
        actions_model = training_model.get_layer('actions_mlp2')(actions_model)

        concat = training_model.get_layer('concat')([state_model, actions_model])

        mlp_output = training_model.get_layer('mlp_output0')(concat)
        mlp_output = training_model.get_layer('mlp_output1')(mlp_output)

        model = Model(inputs=[states_inp, actions_inp], outputs=mlp_output)

        return training_model, model

    def get_reward(self, state, action):
        return self.model.predict([state, action])[0][0]

    def train_model(self, pref_db):
        seg_1s = [triple['seq1'] for triple in pref_db]
        seg_1s = [[[x['state'], x['actions']] for x in seg] for seg in seg_1s]

        seg_2s = [triple['seq2'] for triple in pref_db]
        seg_2s = [[[x['state'], x['actions']] for x in seg] for seg in seg_2s]

        act = []
        obs = []
        for seg1, seg2 in zip(seg_1s, seg_2s):
            act1 = []
            obs1 = []
            act2 = []
            obs2 = []
            for step1, step2 in zip(seg1, seg2):
                act1.append(step1[1])
                act2.append(step2[1])
                obs1.append(step1[0])
                obs2.append(step2[0])
            act1.extend(np.zeros((self.num_steps - len(act1), self.action_size)))
            obs1.extend(np.zeros((self.num_steps - len(obs1), self.state_size)))
            act2.extend(np.zeros((self.num_steps - len(act2), self.action_size)))
            obs2.extend(np.zeros((self.num_steps - len(obs2), self.state_size)))
            act.append([act1, act2])
            obs.append([obs1, obs2])

        prefs = [triple['p'] for triple in pref_db]

        obs = np.array(obs)
        act = np.array(act)
        inputs = [obs, act]
        targets = [prefs, np.zeros((len(prefs),))]

        reward_model_history = self.training_model.fit(inputs, targets, epochs=50, batch_size=30, verbose=0)

        return reward_model_history
