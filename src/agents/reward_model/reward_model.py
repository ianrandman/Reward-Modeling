import numpy as np
import keras.backend as K

from keras.layers import Input, Dense, concatenate, Lambda, Reshape

from keras.optimizers import Adam
from keras.models import Model, Sequential
import math

import tensorflow as tf

class RewardModel:
    def __init__(self, state_size, action_size, num_steps):
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.num_steps = num_steps

        # hyperparameters
        self.lr = 0.001

        # create model
        self.training_model, self.model = self.build_model()

        if self.load_model:
            self.model.load_weights('./save_model/cartpole_actor.h5')

    # def prob_seg_pref(self, seg_1, seg_2):
    #     exp(sum(self.model.predict(seg_1))) / \
    #     (exp(sum(self.model.predict(seg_1))) + exp(sum(self.model.predict(seg_2))))
    #
    # def loss(self, seg1, seg2, pref):
    #     mu_1, mu_2 = self.mu_func(pref)
    #
    #     -sum(mu_1*log(self.prob_seg_pref(seg1, seg2)) +
    #          mu_2*log(self.prob_seg_pref(seg2, seg1)))
    #
    # def mu_func(self, pref):
    #     return pref, 1-pref

    def probability_lambda(self, seg_rewards):
        seg_1_rewards, seg_2_rewards = seg_rewards
        seg_1_rewards = tf.gather(seg_rewards, 0)
        seg_2_rewards = tf.gather(seg_rewards, 1)
        norm_seg_1 = tf.exp(tf.reduce_sum(seg_1_rewards, axis=1))
        norm_seg_2 = tf.exp(tf.reduce_sum(seg_2_rewards, axis=1))
        return tf.divide(norm_seg_1, tf.add(norm_seg_1, norm_seg_2))

    # def loss(self, seg1, seg2, pref):
    #     mu_1, mu_2 = self.mu_func(pref)
    #
        # -sum(mu_1*log(self.prob_seg_pref(seg1, seg2)) +
        #      mu_2*log(self.prob_seg_pref(seg2, seg1)))
    #

    # def mu_func(self, pref):
    #     return pref, 1-pref

    def cust_loss(self, pref, predicted):
        # p_1_over_2, p_2_over_1 = predicted
        p_1_over_2 = tf.gather(predicted, 0)
        p_2_over_1 = tf.gather(predicted, 1)
        mu_1, mu_2 = pref, 1-pref

        return -1 * K.sum(tf.add(tf.multiply(mu_1, tf.log(p_1_over_2)), tf.multiply(mu_2, tf.log(p_2_over_1))))

    def build_model(self):

        def get_mlp():
            state_input = Input(shape=(self.state_size,))
            action_input = Input(shape=(self.action_size,))

            # state branch
            x = Dense(64, activation='relu', kernel_regularizer='l1')(state_input)
            x = Dense(32, activation='relu')(x)
            x = Dense(16, activation='relu')(x)
            x = Model(inputs=state_input, outputs=x)

            # action branch
            y = Dense(16, activation='relu')(action_input)
            y = Dense(32, activation='relu', kernel_regularizer='l1')(y)
            y = Dense(16, activation='relu')(y)
            y = Model(inputs=action_input, outputs=y)

            # combine branches
            combined = concatenate([x.output, y.output])

            # learn after combination
            z = Dense(8, activation='relu')(combined)
            z = Dense(1, activation='tanh')(z)
            model = Model(inputs=[x.input, y.input], outputs=z)

            return model

        model = get_mlp()

        states_input = Input(shape=(2, self.num_steps, self.state_size,))
        # states_reshape = tf.reshape(states_input, [2*self.num_steps, self.state_size])
        states_reshape = Reshape([2*self.num_steps, self.state_size])(states_input)
        actions_input = Input(shape=(2, self.num_steps, self.action_size,))
        # actions_reshape = tf.reshape(actions_input, [2*self.num_steps, self.action_size])
        actions_reshape = Reshape([2*self.num_steps, self.action_size])(actions_input)
        reshape_model =

        ######################

        # state branch
        x1 = Dense(64, activation='relu', kernel_regularizer='l1')(states_reshape)
        x2 = Dense(32, activation='relu')(x1)
        x3 = Dense(16, activation='relu')(x2)

        x_inp = Input(shape=(self.state_size,))
        x_model1 = x1(x_inp)
        x_model2 = x2(x_model1)
        x_model3 = x3(x_model2)
        x_model_final = Model(inputs=x_inp, outputs=x_model3)

        # action branch
        y = Dense(16, activation='relu')(actions_reshape)
        y = Dense(32, activation='relu', kernel_regularizer='l1')(y)
        y = Dense(16, activation='relu')(y)

        y_inp = Input(shape=(self.action_size,))
        y = Model(inputs=actions_reshape, outputs=y)

        # combine branches
        combined = concatenate([x.output, y.output])

        # learn after combination
        z = Dense(8, activation='relu')(combined)
        z = Dense(1, activation='tanh')(z)
        # model = Model(inputs=[x.input, y.input], outputs=z)

        ################

        # state_input = Input(shape=(self.state_size,))
        # action_input = Input(shape=(self.action_size,))

        ###################

        # flat_rewards = model.predict([states_reshape, actions_reshape])
        seg_rewards = tf.reshape(z, [2, self.num_steps])

        seg_1_rewards = tf.gather(seg_rewards, 0)
        seg_2_rewards = tf.gather(seg_rewards, 1)

        p_1_over_2 = Lambda(self.probability_lambda)([seg_1_rewards, seg_2_rewards])
        p_2_over_1 = Lambda(self.probability_lambda)([seg_2_rewards, seg_1_rewards])

        training_model = Model(inputs=[states_input, actions_input], outputs=[p_1_over_2.output, p_2_over_1.output])

        training_model.summary()
        training_model.compile(loss=self.cust_loss, optimizer=Adam(lr=self.lr))
        return training_model, model

    def get_reward(self, state, action):
        return self.model.predict([state], [action]) # TODO state, action

    def train_model(self, pref_db):
        # pref_db = list of triples (triple is a dictionary)
        # for triple in pref_db:
        #     seg_1 = triple['seq1']
        #     seg_2 = triple['seq2']
        #     pref = triple['p']

        seg_1s = [triple['seq1'] for triple in pref_db]
        seg_2s = [triple['seq2'] for triple in pref_db]
        prefs = [triple['p'] for triple in pref_db]

        inputs = np.array([seg_1s, seg_2s])
        targets = np.array(prefs)

        self.training_model.fit(inputs, targets, epochs=50, batch_size=30, verbose=0)
