import torch
from torch import nn
import torch.nn.functional as F

import numpy as np 


class LstmModel(nn.Module):
	"""
	LSTM classifier with two fc layers appended
	"""
	def __init__(self, input_feat_size, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2):
		super(LstmModel, self).__init__()
		self.input_feat_size = input_feat_size
		self.output_size = output_size
		self.hidden_dim = hidden_dim
		self.fc_dim = fc_dim
		self.n_layers = n_layers
		self.dropout = dropout

		self.lstm = nn.LSTM(self.input_feat_size, self.hidden_dim, self.n_layers, dropout = self.dropout, bidirectional = False, batch_first=True)
		self.fc1 = nn.Linear(self.hidden_dim, self.fc_dim)
		self.fc2 = nn.Linear(self.fc_dim, self.output_size)

		self.dp0 = nn.Dropout(p=0.5)
		self.dp1 = nn.Dropout(p=0.5)

	def forward(self, x):
		"""
		Args: x : expected dimention : (batch,seq,feature)
		"""
		batch_size = x.size(0)
		h_0, c_0 = self.init_hidden(batch_size)

		out, hidden = self.lstm(x,(h_0, c_0))
		out = out.contiguous().view(-1, self.hidden_dim)
		out = F.relu(out)
		out = self.dp1(F.relu(self.fc1(out)))
		out = self.fc2(out)
		return out, hidden

	def init_hidden(self, batch_size):
		h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cpu()
		c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cpu()
		return h_0, c_0




class hir_fullModel(nn.Module):
    def __init__(self, batch_size, mode, classifier, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2):
        super(hir_fullModel,self).__init__()
        self.mode = mode
        self.classifier = classifier
        self.audio_lstm = LstmModel(33, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
        self.audio_lstm.load_state_dict(torch.load("./mman/state_dict/audioRNN/audioRNN50.19.pt"))
        for param in self.audio_lstm.parameters():
            param.requires_grad = False
        self.text_lstm = LstmModel(768, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
        self.text_lstm.load_state_dict(torch.load("./mman/state_dict/textRNN/textRNN72.11.pt"))
        for param in self.text_lstm.parameters():
            param.requires_grad = False
        #self.visual_lstm = LstmModel(512, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
        #self.visual_lstm.load_state_dict(torch.load("state_dict/visualRNN/visualRNN56.01.pt"))
        #for param in self.visual_lstm.parameters():
        #    param.requires_grad = False

        if self.mode:
            self.top_lstm = LstmModel(8, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 1)
        else:
#             self.trs_lstm = MAN(batch_size = batch_size, input_feat_size = 150 , audio_dim = 100, visual_dim = 512, text_dim = 100, nhead = 1, nhidden = 300,  nLayers = 1, dropout = 0.5,  output_size = 4, hidden_dim= 300 , fc_dim = 200, n_layers = 1, dropout_lstm = 0.5)
            self.trs_lstm = Transformer_lstmModel( input_feat_size = 150 , audio_dim = 33, text_dim = 768, nhead = 1, nhidden = 300,  nLayers = 1, dropout = 0.5,  output_size = 4, hidden_dim= 300 , fc_dim = 200, n_layers = 1, dropout_lstm = 0.5)
#             self.trs_lstm.load_state_dict(torch.load("test/trs_lstm70.86.pt"))
#             for param in self.trs_lstm.parameters():
#                 param.requires_grad = False
            self.top_lstm = LstmModel(12, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
            self.fc1 = nn.Linear(12, 4)

    def forward(self, audio, text, visual=None):
        batch_size = audio.size()[0]
        audio_s, _ = self.audio_lstm(audio)
        text_s, _ = self.text_lstm(text)
        #visual_s, _ = self.visual_lstm(visual)

        if self.mode:
            #out = torch.cat((audio_s, text_s, visual_s), axis=-1)
            out = torch.cat((audio_s, text_s), axis=-1)
        else:
            fusion_s, _ = self.trs_lstm(audio, text)
            out = torch.cat((audio_s, text_s, fusion_s), axis=-1)
        if self.classifier == 'lstm':
            out = out.view(batch_size, -1, out.size()[-1])
            out, hidden = self.top_lstm(out)
        elif self.classifier == 'mlp':
            out = self.fc1(out)
            hidden = 0
        else:
            out = self.fc1(out)
            hidden = 0

        return out, hidden



class MAN(nn.Module):
	def __init__(self, batch_size, input_feat_size, audio_dim,visual_dim,text_dim, nhead, nhidden, nLayers, dropout, output_size, hidden_dim, fc_dim, n_layers, dropout_lstm):
		super(MAN, self).__init__()
		# Attention
		self.input_feat_size = input_feat_size
		self.audio_dim = audio_dim
		self.visual_dim = visual_dim
		self.text_dim = text_dim
		self.nhead = nhead
		self.nhidden = nhidden
		self.nLayers = nLayers
		self.dropout = dropout
		# LSTM
		self.dropout_lstm = dropout_lstm
		self.output_size = output_size
		self.hidden_dim = hidden_dim
		self.fc_dim = fc_dim
		self.n_layers = n_layers

		self.batch_size = batch_size
	
		self.multimodalattention = multimodalattention(self.batch_size, self.input_feat_size, self.nhidden,self.dropout)

		self.fc_audio = nn.Linear(self.audio_dim, self.input_feat_size)
		self.fc_visual = nn.Linear(self.visual_dim, self.input_feat_size)
		self.fc_text = nn.Linear(self.text_dim, self.input_feat_size)

		self.dropout1 = nn.Dropout(self.dropout)
		self.dropout2 = nn.Dropout(self.dropout)
		self.dropout3 = nn.Dropout(self.dropout)
		
		self.lstm_model = LstmModel(self.input_feat_size*3, output_size = self.output_size, hidden_dim = self.hidden_dim, fc_dim=self.fc_dim, n_layers = self.n_layers, dropout = dropout_lstm)

	def forward(self, audio, text, visual):
		"""
		Args: 
			Audio, visual, text: (batch_size, sequence_length, feature_dimension)
			x: expected input to transformer (Sequence_length, batch_size, input_feature_dimension)
		"""
		# Standadize the audio, text and visual feature dimension
		batch_size = audio.size()[0]

		audio = audio.view(-1, audio.size()[-1])
		text = text.view(-1, text.size()[-1])
		visual = visual.view(-1, visual.size()[-1])

		audio = self.dropout1(F.relu(self.fc_audio(audio)))
		text = self.dropout2(F.relu(self.fc_text(text)))
		visual = self.dropout3(F.relu(self.fc_visual(visual)))

		# audio = self.fc_audio(audio)
		# text = self.fc_text(text)
		# visual = self.fc_visual(visual)

		audio = audio.view(batch_size, -1, self.input_feat_size)
		text = text.view(batch_size, -1, self.input_feat_size)
		visual = visual.view(batch_size, -1, self.input_feat_size)

		# Change the dimension from (Sequence_length, batch_size, input_feature_dimension)
		audio = audio.view(1, -1, self.input_feat_size)
		text = text.view(1, -1, self.input_feat_size)
		visual = visual.view(1, -1, self.input_feat_size)

		# # Concate the modalityies to (batch_size, sequence_length, input_feature_dimension)
		out = torch.cat((audio,  text, visual), axis = 0)

		# attention forward
		out = self.multimodalattention(out)
		out = out.view(out.size()[0], batch_size, -1, out.size()[2])

		audio = out[0,:]
		text = out[1,:]
		visual = out[2,:]

		out = torch.cat((audio, text, visual), axis = -1)
		out, hidden = self.lstm_model(out)

		return out, hidden

class multimodalattention(nn.Module):
	"""docstring for multimodaltransformer"""
	def __init__(self, batch_size, input_feat_size, nhidden, dropout):
		super(multimodalattention, self).__init__()
		self.input_feat_size = input_feat_size
		self.nhidden = nhidden
		self.batch_size = batch_size*110

		self.aq_fc = nn.Linear(input_feat_size, input_feat_size)
		self.vq_fc = nn.Linear(input_feat_size, input_feat_size)
		self.tq_fc = nn.Linear(input_feat_size, input_feat_size)
		self.ak_fc = nn.Linear(input_feat_size, input_feat_size)
		self.vk_fc = nn.Linear(input_feat_size, input_feat_size)
		self.tk_fc = nn.Linear(input_feat_size, input_feat_size)
		self.av_fc = nn.Linear(input_feat_size, input_feat_size)
		self.vv_fc = nn.Linear(input_feat_size, input_feat_size)
		self.tv_fc = nn.Linear(input_feat_size, input_feat_size)

		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		self.softmax = nn.Softmax(dim = 1)

		self.norm1 = nn.LayerNorm(input_feat_size)          # not used
		self.norm2 = nn.LayerNorm(input_feat_size)          # not used

		self.linear1 = nn.Linear(input_feat_size, nhidden)          # not used
		self.dropout = nn.Dropout(dropout)                          # not used
		self.linear2 = nn.Linear(nhidden, input_feat_size)          # not used
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)       					   # not used

	def forward(self,src):
		seq_len, _, feat_dim = src.size()

		# output size [batch_size, dim]
		audio_q = self.aq_fc(src[0,:,:])
		visual_q = self.vq_fc(src[2,:,:])
		text_q = self.tq_fc(src[1,:,:])
		audio_k = self.ak_fc(src[0,:,:])
		visual_k = self.vk_fc(src[2,:,:])
		text_k = self.tk_fc(src[1,:,:])
		audio_v = self.av_fc(src[0,:,:])
		visual_v = self.vv_fc(src[2,:,:])
		text_v = self.tv_fc(src[1,:,:])

		# output size [batch_size, ]
		aa_att = self.cos(audio_q, audio_k).view(-1,1)
		av_att = self.cos(audio_q, visual_k).view(-1,1)
		at_att = self.cos(audio_q, text_k).view(-1,1)
		va_att = self.cos(visual_q, audio_k).view(-1,1)
		vv_att = self.cos(visual_q, visual_k).view(-1,1)
		vt_att = self.cos(visual_q, text_k).view(-1,1)
		tt_att = self.cos(text_q, text_k).view(-1,1)
		ta_att = self.cos(text_q, audio_k).view(-1,1)
		tv_att = self.cos(text_q, visual_k).view(-1,1)

		# output size [batch_size, 3]
		a_att = self.softmax(torch.cat((aa_att, av_att, at_att), axis = 1))
		v_att = self.softmax(torch.cat((va_att, vv_att, vt_att), axis = 1))
		t_att = self.softmax(torch.cat((ta_att, tv_att, tt_att), axis = 1))

		# output size [batch_size, dim]
		a_hat = (a_att[:,0].view(-1,1)*audio_v + a_att[:,0].view(-1,1)*visual_v + a_att[:,0].view(-1,1)*text_v).view(1,-1,self.input_feat_size)
		v_hat = (v_att[:,0].view(-1,1)*audio_v + v_att[:,0].view(-1,1)*visual_v + v_att[:,0].view(-1,1)*text_v).view(1,-1,self.input_feat_size)
		t_hat = (t_att[:,0].view(-1,1)*audio_v + t_att[:,0].view(-1,1)*visual_v + t_att[:,0].view(-1,1)*text_v).view(1,-1,self.input_feat_size)

		# add
		src2 = torch.cat((a_hat, t_hat, v_hat), axis = 0)

		# normalize
		src = src + self.dropout1(src2)

		# src = self.dropout1(src2)
		# src = self.norm1(src)
		# src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
		# src = src + self.dropout2(src2)
		# src = self.norm2(src)

		return src
		
# class maLstmModel(nn.Module):
# 	"""
# 	LSTM classifier with two fc layers appended
# 	"""
# 	def __init__(self, input_feat_size, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2):
# 		super(maLstmModel, self).__init__()
# 		self.input_feat_size = input_feat_size
# 		self.output_size = output_size
# 		self.hidden_dim = hidden_dim
# 		self.fc_dim = fc_dim
# 		self.n_layers = n_layers
# 		self.dropout = dropout

# 		self.lstm = nn.LSTM(self.input_feat_size, self.hidden_dim, self.n_layers, dropout = self.dropout, bidirectional = False, batch_first=True)
# 		self.fc1 = nn.Linear(self.hidden_dim, self.fc_dim)
# 		self.fc2 = nn.Linear(self.fc_dim, self.output_size)

# 		self.dp0 = nn.Dropout(p=0.5)
# 		self.dp1 = nn.Dropout(p=0.5)

# 		self.norm0 = nn.LayerNorm(hidden_dim)
# 		self.norm1 = nn.LayerNorm(fc_dim)


# 	def forward(self, x):
# 		"""
# 		Args: x : expected dimention : (batch,seq,feature)
# 		"""
# 		batch_size = x.size(0)
# 		h_0, c_0 = self.init_hidden(batch_size)

# 		out, hidden = self.lstm(x,(h_0, c_0))
# 		out = out.contiguous().view(-1, self.hidden_dim)
# 		out = self.dp0(self.norm0(F.relu(out)))
# 		out = self.dp1(self.norm1(F.relu(self.fc1(out))))
# 		out = self.fc2(out)
# 		return out, hidden

# 	def init_hidden(self, batch_size):
# 		h_0 = torch.zeros(self.n_layers , batch_size, self.hidden_dim).cuda()
# 		c_0 = torch.zeros(self.n_layers , batch_size, self.hidden_dim).cuda()
# 		return h_0, c_0


class Transformer_lstmModel(nn.Module):
	def __init__(self, input_feat_size, audio_dim, text_dim, nhead, nhidden, nLayers, dropout, output_size, hidden_dim, fc_dim, n_layers, dropout_lstm):
		super(Transformer_lstmModel, self).__init__()
		from torch.nn import TransformerEncoder, TransformerEncoderLayer
		# Transformer
		self.input_feat_size = input_feat_size
		self.audio_dim = audio_dim
		#self.visual_dim = visual_dim
		self.text_dim = text_dim
		self.nhead = nhead
		self.nhidden = nhidden
		self.nLayers = nLayers
		self.dropout = dropout
		# LSTM
		self.dropout_lstm = dropout_lstm
		self.output_size = output_size
		self.hidden_dim = hidden_dim
		self.fc_dim = fc_dim
		self.n_layers = n_layers

		encoder_layers = TransformerEncoderLayer(self.input_feat_size, self.nhead, self.nhidden, self.dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, self.nLayers)

		self.fc_audio = nn.Linear(self.audio_dim, self.input_feat_size)
		#self.fc_visual = nn.Linear(self.visual_dim, self.input_feat_size)
		self.fc_text = nn.Linear(self.text_dim, self.input_feat_size)
		
		self.lstm_model = LstmModel(self.input_feat_size*2, output_size = self.output_size, hidden_dim = self.hidden_dim, fc_dim=self.fc_dim, n_layers = self.n_layers, dropout = dropout_lstm)

	def forward(self, audio, text, visual=None):
		"""
		Args: 
			Audio, visual, text: (batch_size, sequence_length, feature_dimension)
			x: expected input to transformer (Sequence_length, batch_size, input_feature_dimension)
		"""
		# Standadize the audio, text and visual feature dimension
		batch_size = audio.size()[0]

		audio = audio.view(-1, audio.size()[-1])
		text = text.view(-1, text.size()[-1])
		#visual = visual.view(-1, visual.size()[-1])

		audio = F.relu(self.fc_audio(audio))
		text = F.relu(self.fc_text(text))
		#visual = F.relu(self.fc_visual(visual))

		audio = audio.view(batch_size, -1, self.input_feat_size)
		text = text.view(batch_size, -1, self.input_feat_size)
		#visual = visual.view(batch_size, -1, self.input_feat_size)

		# Change the dimension from (Sequence_length, batch_size, input_feature_dimension)
		audio = audio.view(1, -1, self.input_feat_size)
		text = text.view(1, -1, self.input_feat_size)
		#visual = visual.view(1, -1, self.input_feat_size)

		# # Concate the modalityies to (batch_size, sequence_length, input_feature_dimension)
		#out = torch.cat((audio,  text, visual), axis = 0)
		out = torch.cat((audio,  text), axis = 0)

		# Transformer forward
		out = self.transformer_encoder(out)
		out = out.view(out.size()[0], batch_size, -1, out.size()[2])

		audio = out[0,:]
		text = out[1,:]
		#visual = out[2,:]

		#out = torch.cat((audio, text, visual), axis = -1)
		out = torch.cat((audio, text), axis = -1)
		out, hidden = self.lstm_model(out)

		return out, hidden



# class hir_lstmModel(nn.Module):
# 	def __init__(self, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2):
# 		super(hir_lstmModel,self).__init__()
# 		self.audio_lstm = LstmModel(100, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
# 		self.audio_lstm.load_state_dict(torch.load("state_dict/audioRNN/audioRNN57.12.pt"))
# 		for param in self.audio_lstm.parameters():
# 			param.requires_grad = False
# 		self.text_lstm = LstmModel(100, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
# 		self.text_lstm.load_state_dict(torch.load("state_dict/textRNN/textRNN68.95.pt"))
# 		for param in self.text_lstm.parameters():
# 			param.requires_grad = False
# 		self.visual_lstm = LstmModel(512, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 2)
# 		self.visual_lstm.load_state_dict(torch.load("state_dict/visualRNN/visualRNN56.01.pt"))
# 		for param in self.visual_lstm.parameters():
# 			param.requires_grad = False
# 		self.top_lstm = LstmModel(12, output_size, hidden_dim, fc_dim, dropout = 0.5, n_layers = 1)

# 	def forward(self, audio, text, visual):
# 		batch_size = audio.size()[0]
# 		audio, _ = self.audio_lstm(audio)
# 		text, _ = self.text_lstm(text)
# 		visual, _ = self.visual_lstm(visual)
# 		out = torch.cat((audio, text, visual), axis=-1)
# 		out = out.view(batch_size, -1, out.size()[-1])
# 		out, hidden = self.top_lstm(out)
# 		return out, hidden