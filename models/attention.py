import torch
from torch import nn
import torch.nn.functional as F

import random

class Attention(nn.Module):
    
    def __init__(self, query_size,
                       key_size,
                       hidden_size):
        
        super(Attention, self).__init__()
        
        self.key_size = key_size
        self.query_size = query_size
        self.hidden_size = hidden_size
        
        self.query_transform = nn.Linear(self.query_size, self.hidden_size, bias=False) 
        self.key_transform = nn.Linear(self.key_size, self.hidden_size, bias=False)
        
    
    def forward(self, query, keys):
        
        src_seq_len = keys.size(0)
        batch_size = keys.size(1)

        # print(hidden_dec.size())
        query = F.relu(self.query_transform(query))
        keys = F.relu(self.key_transform(keys))

        query = query.unsqueeze(dim=-2)
        keys = keys.unsqueeze(dim=-1)
        e = torch.matmul(query, keys).squeeze(-1)
        # print(e.size(), query.size(), keys.size())
      
        att_weights = F.softmax(e, dim=0)              #[b, src_seq_len]
        
        return att_weights


# class AttentionDecoder(nn.Module):

#     def __init__(self, query_size,
#                        key_size,
#                        att_size,
#                        class_num,
#                        char2idx,
#                        max_len
#                        ):
        
#         super(AttentionDecoder, self).__init__()

#         self.class_num = class_num
#         self.char2idx = char2idx
#         self.max_len = max_len

#         input_size = key_size + hidden_size
#         self.decoder = nn.GRUCell(input_size=input_size,
#                                   hidden_size=query_size)
#         self.attention = Attention(query_size=query_size,
#                                    key_size=key_size,
#                                    hidden_size=att_size)
#         self.emb_dim = dec_hidden_size
#         self.embeddings = nn.Embedding(num_embeddings=len(char2idx), embedding_dim=dec_hidden_size)
#         self.classifier = nn.Linear(dec_hidden_size, self.class_num)
#         self.sos_emb = torch.tensor(char2idx['sos']).cuda()


#     def forward(self, memory, labels):

#         batch_size = memory.size(1)

#         # print('labels', labels.size())
#         labels = labels.permute(1,0)

#         outputs = torch.zeros(self.max_len, batch_size, self.class_num).cuda()
#         dec_input = self.embeddings(self.sos_emb.expand(batch_size, ))
#         query = memory[0]
#         for i in range(max_len):
#             att_weights = self.attention(memory, query)
#             c = torch.mul(att_weights, hidden_enc).sum(dim=0)
#             dec_input = torch.cat((dec_input, c), dim=1)
#             hidden_dec = self.decoder(dec_input, hidden_dec)
#             output = self.classifier(hidden_dec)
#             outputs[i] = output

#             # teacher_force = random.random() < teacher_forcing_ratio
#             teacher_force = True if torch.rand(1).item() < teacher_forcing_ratio else False
#             #get the highest predicted token from our predictions
#             top1 = torch.argmax(output, dim=1)
#             idx = labels[i] if teacher_force else top1
#             #if teacher forcing, use actual next token as next input
#             #if not, use predicted token
#             dec_input = self.embeddings(idx)
        
#         return outputs









        
