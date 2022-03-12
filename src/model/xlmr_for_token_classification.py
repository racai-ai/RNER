from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LateralInhibition import LateralInhibition

class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p,seq_len, label_ignore_idx=0,
                head_init_range=0.04, device='cuda',use_norm=False,use_li=False,li_dropout_p=0.1,li_sigma=10):
        super().__init__()

        self.n_labels = n_labels

        self.seq_len=seq_len
        
        self.use_li=use_li
        self.li_dropout_p=li_dropout_p
        if self.use_li:
            self.li = LateralInhibition(hidden_size, li_sigma)
            if self.li_dropout_p>0:
                self.li_dropout = nn.Dropout(li_dropout_p)

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.classification_head = nn.Linear(hidden_size, n_labels)
        
        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model

        self.use_norm=use_norm
        if self.use_norm:
            self.norm=nn.BatchNorm1d(seq_len)
        
        self.device=device

        # initializing classification head
        self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)

    def forward(self, inputs_ids, labels, labels_mask, valid_mask):
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
            transformer_out=self.norm(transformer_out)

        out_1 = F.relu(self.linear_1(transformer_out))
        out_1 = self.dropout(out_1)
        logits = self.classification_head(out_1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            return loss
        else:
            return logits

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]
