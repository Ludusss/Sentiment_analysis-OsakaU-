import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP_2(nn.Module):
    def __init__(self, input_feature_size, hidden_size, fc_dim, n_classes, n_layers, device):
        super(MLP_2, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device
        self.fc_dim = fc_dim

        self.fc = nn.Linear(self.input_feature_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.fc_dim, self.n_classes)

    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class MLP(nn.Module):
    def __init__(self, input_feature_size, hidden_size, n_classes, n_layers, device):
        super(MLP, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device

        self.fc = nn.Linear(self.input_feature_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)

        return out


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


class LSTMSep(nn.Module):
    def __init__(self, input_feature_size_text, input_feature_size_audio, hidden_size, n_classes, fc_dim, n_layers, device):
        super(LSTMSep, self).__init__()

        self.text_lstm = LSTM(input_feature_size_text, hidden_size, n_classes, fc_dim, n_layers, device)
        self.audio_lstm = LSTM(input_feature_size_audio, hidden_size, n_classes, fc_dim, n_layers, device)

    def forward(self, text, audio):
        text_out = self.text_lstm(text)
        audio_out = self.audio_lstm(audio)

        return text_out, audio_out


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
        # out : (batch_size, seq_length, hidden_size()
        out = out.contiguous().view(-1, self.hidden_size)
        # out : (batch_size, hidden_size)
        out = self.fc(torch.sigmoid(out))
        return out


class LSTM_ATTN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, n_classes, device,
                 n_layers=1, use_bidirectional=False, use_dropout=False):
        super(LSTM_ATTN, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device

        self.lstm = nn.LSTM(input_feature_size, hidden_size, n_layers,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0., batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention(self, lstm_output, final_state):
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        print(weights.size())
        print(lstm_output.permute(0,2,1).size())
        return torch.bmm(lstm_output, weights)

    def forward(self, x):
        # x -> (batch_size, seq_length, feature_size)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        output, (hidden, _) = self.lstm(x, (h0, c0))


        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(output, hidden)
        print(attn_output.size())
        attn_output = attn_output.contiguous().view(-1, self.hidden_size)

        return self.fc(attn_output.squeeze(0))
