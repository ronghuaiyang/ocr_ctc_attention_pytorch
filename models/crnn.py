import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .resnet import resnet18, resnet34
from .densenet import densenet18,densenet121
from .shufflenet import shufflenet_v2_x1_0
from .attention import Attention

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

        self.class_num = class_num
        # self.cnn = densenet18()
        self.cnn = resnet18()
        # self.cnn = shufflenet_v2_x1_0()
        d_model = 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.5)
        self.fc_classify = nn.Linear(d_model, self.class_num)

    def forward(self, input):
        x = self.cnn(input)
        # print(x.size())
        # nchw->ncw
        x = x.squeeze(2)
        # ncw->wnc w:t
        x = x.permute(2, 0, 1)
        cnn_features = x

        # use ctc
        x = self.pos_encoder(x)
        x = self.encoder(x)
        outputs = self.fc_classify(x)

        return outputs, cnn_features

class TransformerDecoder(nn.Module):

    def __init__(self, d_model, class_num, max_len, char2idx=None):
        super(Decoder, self).__init__()

        # self.tgt_mask = self.transform.generate_square_subsequent_mask(200)
        # self.d_model = d_model
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.max_len = max_len
        self.embeddings = nn.Embedding(num_embeddings=len(char2idx), embedding_dim=d_model)
        self.sos_emb = torch.tensor(char2idx['sos']).cuda()
        self.tgt_mask = nn.Transformer().generate_square_subsequent_mask(max_len).cuda()
        self.predictor = nn.Linear(d_model, len(char2idx))

    def forward(self, input, labels=None):
        # input_size T,N,D
        batch_size = input.size(1)

        if self.training:
            labels = labels.permute(1,0)
            sos_emb = self.sos_emb.expand(1,batch_size)
            labels = torch.cat((sos_emb, labels), dim=0)[:self.max_len]
            tgt_emb = self.embeddings(labels)
            out = self.decoder(tgt=tgt_emb,
                               tgt_mask=self.tgt_mask,
                               memory=input)
            return self.predictor(out)
        else:
            labels = self.sos_emb.expand(1, batch_size)
            outputs = torch.zeros(self.max_len, batch_size, self.class_num).cuda()
            for i in range(0, self.max_len):
                tgt_emb = self.embeddings(labels)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_mask = nn.Transformer().generate_square_subsequent_mask(i+1).cuda()
                out = self.decoder(tgt=tgt_emb, tgt_mask=tgt_mask, memory=input)           
                out = self.predictor(out)
                pred = torch.argmax(out, dim=2)
                sos = self.sos_emb.expand(1, batch_size)
                labels = torch.cat((sos, pred), dim=0)
                outputs[:i] = out
            return outputs


class RNNAttentionDecoder(nn.Module):

    def __init__(self, d_model,
                       class_num,
                       max_len,
                       char2idx,
                       ):
        
        super(RNNAttentionDecoder, self).__init__()

        self.class_num = class_num
        self.char2idx = char2idx
        self.max_len = max_len
        # self.teacher_forcing_ratio = teacher_forcing_ratio

        input_size = d_model + d_model
        self.decoder = nn.GRUCell(input_size=input_size,
                                  hidden_size=d_model)
        self.attention = Attention(query_size=d_model,
                                   key_size=d_model,
                                   hidden_size=d_model)
        self.embeddings = nn.Embedding(num_embeddings=len(char2idx), embedding_dim=d_model)
        self.classifier = nn.Linear(d_model, self.class_num)
        self.sos_emb = torch.tensor(char2idx['sos']).cuda()


    def forward(self, inputs, labels=None):

        batch_size = inputs.size(1)

        # if self.training:
        outputs = torch.zeros(self.max_len, batch_size, self.class_num).cuda()
        query = inputs[0]
        keys = inputs
        out = self.embeddings(self.sos_emb.expand(batch_size, ))
        if self.training: labels = labels.permute(1,0)
        for i in range(self.max_len):
            att_weights = self.attention(inputs, keys)
            c = torch.mul(att_weights, keys).sum(dim=0)
            input = torch.cat((out, c), dim=1)
            query = self.decoder(input, query)
            pred = self.classifier(query)
            outputs[i] = pred
            if self.training:
                out = self.embeddings(labels[i])
            else:
                top1 = torch.argmax(pred, dim=1)
                out = self.embeddings(top1)             
        return outputs


class AttentionHead(nn.Module):

    def __init__(self, class_num, max_len, char2idx=None):
        super(AttentionHead, self).__init__()

        # encoder
        d_model = 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.5)

        # decoder
        # self.decoder = TransformerDecoder(d_model=d_model,
        #                                   class_num=class_num,
        #                                   max_len=max_len,
        #                                   char2idx=char2idx)

        self.decoder = RNNAttentionDecoder(d_model=d_model,
                                           class_num=class_num,
                                           max_len=max_len,
                                           char2idx=char2idx)

    def forward(self, input, labels=None):
        # input_size T,N,D
        x = self.pos_encoder(input)
        x = self.encoder(x)
        out = self.decoder(x, labels)
        return out

