import collections
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import random
import math
from .DuelingNet import *


class DoubleDuelingDQN():
    def __init__(self, memory_capacity, TRARGE_REPLACE_INTER, batch_size, n_states, n_actions, lr, device, epsilon,
                 epsilon_min, epsilon_decay, gamma):
        self.memory_capacity = memory_capacity
        self.TRARGE_REPLACE_INTER = TRARGE_REPLACE_INTER
        self.batch_size = batch_size
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.device = device
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.eval_net = DuelingNet(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.target_net = DuelingNet(n_states=self.n_states, n_actions=self.n_actions).to(self.device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = torch.zeros([self.memory_capacity, self.n_states * 2 + 2])
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.index = np.arange(self.memory_capacity)

        self.sample_count = 0

        # UCB parameters
        self.action_counts = np.zeros(self.n_actions)
        self.total_count = 0
        self.ucb_c = 1.0  # UCB探索因子

        # Boltzmann parameters
        self.temperature = 1.0  # 温度参数
        self.min_temperature = 0.1  # 最小温度
        self.temperature_decay = 0.99  # 温度衰减率

    def choose_action(self, state, strategy='epsilon'):
        if strategy == 'ucb':
            action = self.ucb_action_selection(state)
        elif strategy == 'boltzmann':
            action = self.boltzmann_action_selection(state)
        elif strategy == 'entropy':
            action = self.entropy_based_action_selection(state)
        elif strategy == 'noise':
            action = self.noise_based_action_selection(state)
        else:
            action = self.epsilon_greedy_action_selection(state)
        return action

    def epsilon_greedy_action_selection(self, state):
        if np.random.uniform() < self.epsilon:
            action = torch.randint(0, self.n_actions, (state.shape[0],), device=self.device)
        else:
            actions_value = self.eval_net(state)
            action = torch.argmax(actions_value, dim=1).cpu()
        self.update_epsilon()
        return action

    def ucb_action_selection(self, state):
        self.total_count += 1
        actions_value = self.eval_net(state)
        ucb_values = np.zeros((state.shape[0], self.n_actions))

        for i in range(state.shape[0]):
            for a in range(self.n_actions):
                if self.action_counts[a] > 0:
                    ucb_values[i, a] = actions_value[i, a].item() + self.ucb_c * np.sqrt(
                        np.log(self.total_count) / self.action_counts[a])
                else:
                    ucb_values[i, a] = float('inf')  # 保证每个动作至少被选择一次

        action = torch.argmax(torch.tensor(ucb_values), dim=1).cpu()
        for a in action:
            self.action_counts[a] += 1

        return action

    def boltzmann_action_selection(self, state):
        actions_value = self.eval_net(state)
        probabilities = F.softmax(actions_value / self.temperature, dim=1).cpu().detach().numpy()
        action = np.array([np.random.choice(self.n_actions, p=prob) for prob in probabilities])
        self.update_temperature()
        return torch.tensor(action)

    def update_temperature(self):
        if self.temperature > self.min_temperature:
            self.temperature *= self.temperature_decay
        else:
            self.temperature = self.min_temperature

    def entropy_based_action_selection(self, state):
        actions_value = self.eval_net(state)
        probabilities = F.softmax(actions_value, dim=1).cpu().detach().numpy()
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-6), axis=1)
        action = np.argmax(entropy)
        return torch.tensor(action)

    def noise_based_action_selection(self, state):
        actions_value = self.eval_net(state)
        noise = torch.randn_like(actions_value) * 0.1  # 添加噪声
        actions_value += noise
        action = torch.argmax(actions_value, dim=1).cpu()
        return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def test(self, x):
        x = torch.squeeze(x)
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1]  # 返回最大Q值对应的动作
        return action

    def store_transition(self, s, a, r, s_):
        """存放经验"""
        transition = torch.cat((s, a, r, s_), dim=1)

        _l = transition.size(0)
        _index = np.arange(self.memory_counter, self.memory_counter + _l) % self.memory_capacity

        self.memory_counter += _l
        self.memory[_index, :] = transition.cpu()

    def learn(self):
        if self.learn_step_counter % self.TRARGE_REPLACE_INTER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 更新目标网络

        _index = np.random.choice(self.index, self.batch_size, replace=False)
        batch_sample = self.memory[_index, :]

        # 从经验回放池中随机采样
        b_s = Variable(batch_sample[:, :self.n_states].type(torch.FloatTensor).to(self.device))
        b_a = Variable(batch_sample[:, self.n_states:self.n_states + 1].type(torch.LongTensor).to(self.device))
        b_r = Variable(batch_sample[:, self.n_states + 1:self.n_states + 2].type(torch.FloatTensor).to(self.device))
        b_s_ = Variable(batch_sample[:, -self.n_states:].type(torch.FloatTensor).to(self.device))

        q_values = self.eval_net(b_s)  # shape (batch, 1)
        next_q_values = self.eval_net(b_s_)
        next_q_state_values = self.target_net(b_s_)
        q_value = q_values.gather(1, b_a)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))
        expected_q_value = b_r + self.gamma * next_q_value
        loss = self.loss_func(q_value, expected_q_value.data.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
