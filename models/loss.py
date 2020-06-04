import torch
import torch.nn as nn

class CTCFocalLoss(nn.Module):
    '''
    p = e^(-ctc_loss)
    focal_loss = alpha*(1-p)^gamma*ctc_loss
    '''

    def __init__(self, gamma, blank):

        super(CTCFocalLoss, self).__init__()

        self.gamma = gamma
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none')

    def forward(self, input, label, input_length, labels_length):

        ctc_loss = self.ctc_loss(input, label, input_length, labels_length)
        p = torch.exp(ctc_loss*(-1))
        loss = ((1 - p)**self.gamma)*ctc_loss
        loss = loss / labels_length
        loss = loss.mean()
        return loss
        
    





    