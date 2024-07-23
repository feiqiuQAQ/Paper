import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output).squeeze(2)  # (batch_size, seq_len)
        soft_attn_weights = F.softmax(attn_weights, 1)  # (batch_size, seq_len)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output)  # (batch_size, 1, hidden_size)
        return context, soft_attn_weights
