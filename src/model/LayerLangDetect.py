from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LateralInhibition import LateralInhibition

class LayerLangDetect(nn.Module):

    def __init__(self, n_labels, hidden_size, dropout_p, head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels
        
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.classification_head = nn.Linear(hidden_size, n_labels)
        
        self.device=device

        # initializing classification head
        self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)

    def forward(self, transformer_out):
        '''
        Computes a forward pass through the sequence tagging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where 
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''

        # Use first token for classification
        out_1 = F.relu(self.linear_1(transformer_out[:,0,:]))
        out_1 = self.dropout(out_1)
        logits = self.classification_head(out_1)
        return logits

