import numpy as np
from torch import nn
import torch

from encoder.params_model import *
from encoder.params_data import *
from encoder.data_objects.iemocap_dataset import emo_categories


class EmoEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)

        self.linear_cls = nn.Linear(in_features=model_embedding_size,
                                    out_features=len(emo_categories)).to(device)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)

        pred = self.linear_cls(embeds)

        return embeds, pred


class StackedBiLSTMEmoEncoder(nn.Module):
    def __init__(self, device):
        super(StackedBiLSTMEmoEncoder, self).__init__()
        self.device = device

        self.lstm1 = nn.LSTM(input_size=mel_n_channels,
                             hidden_size=512,
                             bidirectional=True,
                             batch_first=True).to(device)

        self.lstm2 = nn.LSTM(input_size=1024,
                             hidden_size=256,
                             bidirectional=True,
                             batch_first=True).to(device)

        self.linear = nn.Linear(in_features=512,
                                out_features=512).to(device)
        self.tanh = nn.Tanh().to(device)

        self.linear_cls = nn.Linear(in_features=512,
                                    out_features=len(emo_categories)).to(device)

    def forward(self, utterances, hidden_init=None):
        o, _ = self.lstm1(utterances, hidden_init)

        o, (h, c) = self.lstm2(o)

        # Take the hidden state of last layers and concatenate the two directions
        h = torch.transpose(h[-2:], 0, 1)
        h = h.reshape([h.shape[0], -1])
        embeds = self.tanh(self.linear(h))

        pred = self.linear_cls(embeds)

        return embeds, pred
