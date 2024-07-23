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


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector


class DuelingNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DuelingNet, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(n_states, 300),
            nn.ReLU()
        )

        self.attention = Attention(300)

        self.advantage = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.shared_layers(x)
        # x = self.attention(x.unsqueeze(1))  # Apply attention mechanism
        # x = x.squeeze(1)  # Squeeze the extra dimension added by attention
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()