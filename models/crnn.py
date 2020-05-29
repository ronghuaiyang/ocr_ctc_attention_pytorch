import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .resnet import resnet18

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CRNN(nn.Module):

    def __init__(self, class_num):
        super(CRNN, self).__init__()

        self.cnn = resnet18()
        # self.rnn = nn.LSTM(input_size=512,
        #                     hidden_size=256,
        #                     num_layers=2,
        #                     dropout=0.5,
        #                     bidirectional=True)
        self.rnn = nn.GRU(input_size=512,
                            hidden_size=256,
                            num_layers=2,
                            dropout=0.5,
                            bidirectional=True)
        self.transform = nn.Transformer(d_model=512,
                                        nhead=8,
                                        num_encoder_layers=1,
                                        num_decoder_layers=1,
                                        dim_feedforward=2048,
                                        dropout=0.5)
        self.tgt_mask = self.transform.generate_square_subsequent_mask(100)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_encoder = PositionalEncoding(d_model=512, dropout=0.5, max_len=100)
        self.fc = nn.Linear(512, class_num)

    # def gen_mask(self, batch_size, target_length, max_length):
    #     mask = torch.range(0, max_length-1)
    #     mask = mask.expand(batch_size, -1)
    #     target_length = target_length.expand(batch_size, max_length)
    #     print(target_length)
    #     return mask



    def forward(self, input):

        x = self.cnn(input)
        # nchw->ncw
        x = x.squeeze(2)
        # ncw->wnc w:t
        x = x.permute(2, 0, 1)

        # use rnn
        # x, _ = self.rnn(x)

        # use transformer
        # x = self.pos_encoder(x)
        # # x = self.transform(x, target, tgt_mask=self.tgt_mask)
        # x = self.encoder(x)
        # print('x shape', x.size())

        output = self.fc(x)
        # print('output shape', output.size())
        # output = F.log_softmax(x, dim=1)
        return output