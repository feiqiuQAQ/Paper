import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .masked_mse import *
from .attention import Attention


class AutomaticWeightedLoss(nn.Module):
    """自动加权多任务损失"""

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class MyModelWithDynamicWeights(nn.Module):
    def __init__(self, input_size, hidden_size, num_features):
        super(MyModelWithDynamicWeights, self).__init__()
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

        # Uncertainty parameters
        self.awl = AutomaticWeightedLoss(2)  # 自动加权损失

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
        state_loss = MaskedMSE()(state_prediction, b_y[:, :, :self.num_features], state_mask)
        reward_loss = MaskedMSE()(reward_prediction, b_y[:, :, -1].unsqueeze(2), reward_mask)
        loss = self.awl(state_loss, reward_loss)
        return loss
