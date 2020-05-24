from pdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init(module, weight_init, bias_init, gain=1):
    """

    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class A2CNet(nn.Module):
    def __init__(self, obs_size, num_actions, writer=None):
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param n_stack: number of frames stacked
        :param num_actions: size of the action space, pass env.action_space.n
        :param in_size: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        self.writer = writer

        # constants
        self.obs_size = obs_size
        self.num_actions = num_actions

        # networks
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        # self.feat_enc_net = FeatureEncoderNet(n_stack, self.in_size)
        self.actor = init_(nn.Linear(self.obs_size, self.num_actions))  # estimates what to do
        self.critic = init_(nn.Linear(self.obs_size,
                                      1))  # estimates how good the value function (how good the current state is)

    def forward(self, state):
        """

        feature: current encoded state

        :param state: current state
        :return:
        """

        # encode the state
        feature = self.feat_enc_net(state)

        # calculate policy and value function
        policy = self.actor(feature)
        value = self.critic(feature)

        if self.writer is not None:
            self.writer.add_histogram("feature", feature.detach())
            self.writer.add_histogram("policy", policy.detach())
            self.writer.add_histogram("value", value.detach())

        return policy, torch.squeeze(value), feature

    def get_action(self, state):
        """
        Method for selecting the next action

        :param state: current state
        :return: tuple of (action, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policy, value, feature = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return (action, cat.log_prob(action), cat.entropy().mean(), value,
                feature)
