import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

class LoadableModule(torch.nn.Module):
    def load(self, model_path):
        try:
            super(LoadableModule, self).load_state_dict()
        except:
            print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
                model_path, get_device()))
            super(LoadableModule, self).load_state_dict(torch.load(model_path, map_location=get_device()))

    def forward(self, input):
        raise Exception("Not implemented!")

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
    def __init__(self, input_feature_size, hidden_size, n_classes, n_layers, device, p=0.2554070776341251):
        super(MLP, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device

        self.fc = nn.Linear(self.input_feature_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)

        return out

class AttentionLSTM(LoadableModule):
    """Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self, cfg):
        """
        LSTM with self-Attention model.
        :param cfg: Linguistic config object
        """
        super(AttentionLSTM, self).__init__()
        self.batch_size = cfg.batch_size
        self.output_size = cfg.num_classes
        self.hidden_size = cfg.hidden_dim
        self.embedding_length = cfg.emb_dim

        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.dropout2 = torch.nn.Dropout(cfg.dropout2)

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, batch_first=True)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """
        This method computes soft alignment scores for each of the hidden_states and the last hidden_state of the LSTM.
        Tensor Sizes :
            hidden.shape = (batch_size, hidden_size)
            attn_weights.shape = (batch_size, num_seq)
            soft_attn_weights.shape = (batch_size, num_seq)
            new_hidden_state.shape = (batch_size, hidden_size)
        :param lstm_output: Final output of the LSTM which contains hidden layer outputs for each sequence.
        :param final_state: Final time-step hidden state (h_n) of the LSTM
        :return: Context vector produced by performing weighted sum of all hidden states with attention weights
        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def extract(self, input):
        #input = input.transpose(0, 1)
        input = self.dropout(input)

        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        #output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def classify(self, attn_output):
        attn_output = self.dropout2(attn_output)
        logits = self.label(attn_output)
        return logits.squeeze(1)

    def forward(self, input):
        attn_output = self.extract(input)
        logits = self.classify(attn_output)
        return logits

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
