import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .masked_mse import *
from .attention import Attention


class MyModelWithDWA(nn.Module):
    def __init__(self, input_size, hidden_size, num_features):
        super(MyModelWithDWA, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.shared_layer = nn.Linear(hidden_size, hidden_size)  # 共享特征层
        self.state_attention = Attention(hidden_size)
        self.reward_attention = Attention(hidden_size)
        self.state_layer = nn.Linear(hidden_size, self.num_features)
        self.reward_layer = nn.Linear(hidden_size, 1)

        # DWA weights
        self.alpha = nn.Parameter(torch.ones(2))

    def forward(self, x, x_lengths, h_state):
        x = pack_padded_sequence(x, x_lengths, batch_first=True)
        r_out, h_state = self.lstm(x, h_state)
        padded_out, padded_lengths = pad_packed_sequence(r_out, batch_first=True)
        shared_features = self.shared_layer(padded_out)  # 共享特征
        state_outs = []
        reward_outs = []
        for time_step in range(padded_out.size(1)):  # calculate output for each time step
            state_context, _ = self.state_attention(shared_features[:, :time_step + 1, :])
            reward_context, _ = self.reward_attention(shared_features[:, :time_step + 1, :])
            state_outs.append(self.state_layer(state_context.squeeze(1)))
            reward_outs.append(self.reward_layer(reward_context.squeeze(1)))

        return torch.stack(state_outs, dim=1), torch.stack(reward_outs, dim=1), h_state

    def loss(self, state_prediction, reward_prediction, b_y, state_mask, reward_mask):
        state_loss = MaskedMSE()(state_prediction, b_y[:, :, :11], state_mask)
        reward_loss = MaskedMSE()(reward_prediction, b_y[:, :, -1].unsqueeze(2), reward_mask)
        weights = F.softmax(self.alpha, dim=0)
        loss = weights[0] * state_loss + weights[1] * reward_loss
        return loss
