from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LateralInhibition import LateralInhibition

class LayerXLMREmbeddings(nn.Module):

    def __init__(self, pretrained_path, hidden_size, device='cuda',use_norm=False,
                 use_li=False,li_dropout_p=0.1,li_sigma=10):
        super().__init__()

        self.use_li=use_li
        self.li_dropout_p=li_dropout_p
        if self.use_li:
            self.li = LateralInhibition(hidden_size, li_sigma)
            if self.li_dropout_p>0:
                self.li_dropout = nn.Dropout(li_dropout_p)

        self.use_norm=use_norm
        if use_norm:
            self.norm=nn.BatchNorm1d(hidden_size)

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        
        self.device=device

    def forward(self, inputs_ids):
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
        transformer_out, _ = self.model(inputs_ids, features_only=True)

        if self.use_li:
            transformer_out = self.li(transformer_out)
            if self.li_dropout_p>0:
                transformer_out = self.li_dropout(transformer_out)

        if self.use_norm:
            transformer_out = self.norm(transformer_out)

        return transformer_out

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]
