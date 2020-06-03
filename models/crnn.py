import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .resnet import resnet18
from .attention import Attention, AttentionDecoder

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

    def __init__(self, class_num, char2idx):
        super(CRNN, self).__init__()

        self.class_num = class_num
        self.cnn = resnet18()
        # self.rnn = nn.LSTM(input_size=512,
        #                     hidden_size=256,
        #                     num_layers=2,
        #                     dropout=0.5,
        #                     bidirectional=True)
        # self.rnn = nn.GRU(input_size=512,
        #                     hidden_size=256,
        #                     num_layers=2,
        #                     dropout=0.5,
        #                     bidirectional=True)
        # self.transform = nn.Transformer(d_model=512,
        #                                 nhead=8,
        #                                 num_encoder_layers=1,
        #                                 num_decoder_layers=1,
        #                                 dim_feedforward=2048,
        #                                 dropout=0.5)
        # self.tgt_mask = self.transform.generate_square_subsequent_mask(200)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.pos_encoder = PositionalEncoding(d_model=512, dropout=0.5, max_len=200)
        # if use bidirection, decoder input_size should be 2*encoder_hidden_size
        # self.encoder = nn.GRU(input_size=rnn_params['input_size'],
        #                       hidden_size=rnn_params['hidden_size'],
        #                       bidirectional=rnn_params['bidirectional'])
        # self.decoder = nn.GRUCell(input_size=rnn_params['input_size'],
        #                           hidden_size=rnn_params['hidden_size'])
        # self.attention = Attention(enc_hidden_size=rnn_params['hidden_size'],
        #                            dec_hidden_size=rnn_params['hidden_size'],
        #                            hidden_size=rnn_params['hidden_size'])
        # self.fc_classify = nn.Linear(256, class_num)
        # self.fc_decoder = nn.Linear(256, 256)
        # self.max_target_len = max_target_len
        # self.sos_emb = torch.zeros(256).cuda()

        # use attention need concat hidden_state and contex_state as rnn cell input
        # the input_size shoud be hidden_size * 2
        rnn_params = {'input_size':512, 'hidden_size':256}
        self.fc = nn.Linear(512, rnn_params['hidden_size'])
        self.decoder = AttentionDecoder(input_size=rnn_params['input_size'],
                                        enc_hidden_size=rnn_params['hidden_size'],
                                        dec_hidden_size=rnn_params['hidden_size'],
                                        att_hidden_size=rnn_params['hidden_size'],
                                        class_num=class_num,
                                        char2idx=char2idx
                                        )

    # def gen_mask(self, batch_size, target_length, max_length):
    #     mask = torch.range(0, max_length-1)
    #     mask = mask.expand(batch_size, -1)
    #     target_length = target_length.expand(batch_size, max_length)
    #     print(target_length)
    #     return mask



    def forward(self, input, labels, teacher_forcing_ratio=0.5):
        batch_size = input.size(0)
        x = self.cnn(input)
        # nchw->ncw
        x = x.squeeze(2)

        # ncw->wnc w:t
        x = x.permute(2, 0, 1)
        x = F.relu(self.fc(x))
        # print(x.size())

        h_enc, h_dec = x, x[-1]

        outputs = self.decoder(h_enc, h_dec, labels, teacher_forcing_ratio)

        # use rnn
        # x, _ = self.rnn(x)

        # use transformer
        # x = self.pos_encoder(x)
        # # x = self.transform(x, target, tgt_mask=self.tgt_mask)
        # h_enc, h_dec = self.encoder(x)
        # h_dec = h_dec.squeeze()
        # print(h_enc.size(), h_dec.size())
        # h_dec = torch.cat((h_dec[0], h_dec[1]), dim=1)
        # outputs = torch.zeros(self.max_target_len, batch_size, self.class_num).cuda()
        # target = self.sos_emb.expand(batch_size, -1)
        # for i in range(self.max_target_len):
        #     att_weights = self.attention(h_enc, h_dec).unsqueeze(dim=2)
        #     c = torch.mul(att_weights, h_enc).sum(dim=0)
        #     # print(target.size(), c.size())
        #     gru_input = torch.cat((target, c), dim=1)
        #     h_dec = self.decoder(gru_input, h_dec)
        #     target = F.relu(self.fc_decoder(h_dec))
        #     outputs[i,:,:] = self.fc_classify(target)

        # outputs = self.fc_classify(outputs)
        # print('output shape', output.size())
        # outputs = F.log_softmax(outputs, dim=1)
        return outputs