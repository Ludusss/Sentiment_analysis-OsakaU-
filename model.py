import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_feature_size, hidden_size, n_classes, fc_dim, n_layers, device):
        super(LSTM, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device
        self.fc_dim = fc_dim

        self.lstm = nn.LSTM(self.input_feature_size, self.hidden_size, self.n_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.fc_dim)
        self.fc1 = nn.Linear(self.fc_dim, self.n_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x -> (batch_size, seq_length, feature_size)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        # out : (batch_size, seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        # out : (N,  200)
        out = self.dropout(F.relu(self.fc(out)))
        # out : (N, fc_dim)
        out = self.fc1(out)
        # out : (N, n_classes)
        return out


class LSTM1(nn.Module):
    def __init__(self, input_feature_size, hidden_size, n_classes, n_layers, device):
        super(LSTM1, self).__init__()
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
        out = self.fc(torch.sigmoid(out))
        return out

            