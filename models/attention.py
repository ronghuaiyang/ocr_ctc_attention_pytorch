import torch
from torch import nn
import torch.nn.functional as F

import random

class Attention(nn.Module):
    
    def __init__(self, enc_hidden_size,
                       dec_hidden_size,
                       hidden_size):
        
        super(Attention, self).__init__()
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.hidden_size = hidden_size
        
        self.query_transform = nn.Linear(self.dec_hidden_size, self.hidden_size, bias=False) 
        self.key_transform = nn.Linear(self.enc_hidden_size, self.hidden_size, bias=False)
        
    
    def forward(self, hidden_enc, hidden_dec):
        
        src_seq_len = hidden_enc.size(0)
        batch_size = hidden_enc.size(1)

        # print(hidden_dec.size())
        query = F.relu(self.query_transform(hidden_dec))
        keys = F.relu(self.key_transform(hidden_enc))

        query = query.unsqueeze(dim=-2)
        keys = keys.unsqueeze(dim=-1)

        e = torch.matmul(query, keys).squeeze()
        # print(e.size(), query.size(), keys.size())
      
        att_weights = F.softmax(e, dim=0)              #[b, src_seq_len]
        
        return att_weights


class AttentionDecoder(nn.Module):

    def __init__(self, input_size,
                       enc_hidden_size,
                       dec_hidden_size,
                       att_hidden_size,
                       class_num,
                       char2idx,
                       ):
        
        super(AttentionDecoder, self).__init__()

        self.class_num = class_num
        self.char2idx = char2idx

        self.decoder = nn.GRUCell(input_size=input_size,
                                  hidden_size=dec_hidden_size)
        self.attention = Attention(enc_hidden_size=enc_hidden_size,
                                   dec_hidden_size=dec_hidden_size,
                                   hidden_size=att_hidden_size)
        self.embeddings = nn.Embedding(num_embeddings=len(char2idx), embedding_dim=dec_hidden_size).cuda()
        self.classifier = nn.Linear(dec_hidden_size, self.class_num)

    def forward(self, hidden_enc,
                      hidden_dec,
                      labels,
                      teacher_forcing_ratio):

        # print('labels', labels.size())
        labels = labels.permute(1,0)

        max_len = labels.size(0)
        batch_size = labels.size(1)
        outputs = torch.zeros(max_len, batch_size, self.class_num).cuda()
        dec_input = self.embeddings(torch.zeros(batch_size).long().cuda() + self.char2idx['sos'])
        for i in range(max_len):
            att_weights = self.attention(hidden_enc, hidden_dec).unsqueeze(dim=2)
            c = torch.mul(att_weights, hidden_enc).sum(dim=0)
            # print('dec_input', labels[i].size(), target.size(), c.size())
            dec_input = torch.cat((dec_input, c), dim=1)
            hidden_dec = self.decoder(dec_input, hidden_dec)
            # print('hidden_dec', hidden_dec.size())
            output = self.classifier(hidden_dec)
            outputs[i] = output

            # teacher_force = random.random() < teacher_forcing_ratio
            teacher_force = True if torch.rand(1).item() < teacher_forcing_ratio else False
            #get the highest predicted token from our predictions
            top1 = torch.argmax(output, dim=1)
            idx = labels[i] if teacher_force else top1
            # idx = top1
            # print(idx)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            dec_input = self.embeddings(idx)
        
        return outputs









        
