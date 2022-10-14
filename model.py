import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_feature_size, hidden_size, n_classes, n_layers, device):
        super(LSTM, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device

        self.lstm = nn.LSTM(self.input_feature_size, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, x):
        # x -> (batch_size, seq_length, feature_size)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        # out : (batch_size, seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        # out : (batch_size, hidden_size)
        out = self.fc(out)
        return out

            