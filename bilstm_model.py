"""
Author: Jude Park
"""

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import torch


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, padding_idx:int, n_classes: int,
                 pretrained_embedding: torch.Tensor, dr_rate: float) -> None:
        super().__init__()
        self.dr_rate = dr_rate
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.dr_rate = dr_rate

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx)
        self.embed.weight.data.copy_(pretrained_embedding)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_size,
                            bidirectional=True,
                            dropout=dr_rate)

        self.dropout = nn.Dropout(dr_rate)

        self.fc = nn.Linear(hidden_size * 2, n_classes)


    def forward(self, seq, length):
        """
        :param seq: [max_seq_len * bs]
        :param length: [lengths...]
        :return:
        """

        embed = self.embed(seq) # [max_seq_len * bs * embed_dim]
        packed_input = pack_padded_sequence(embed, length)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        #
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)
