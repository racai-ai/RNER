from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LayerXLMREmbeddings import LayerXLMREmbeddings
from model.LayerLangDetect import LayerLangDetect

class ModelLangDetect(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p,
                 head_init_range=0.04, device='cuda', use_norm=False, use_li=False, li_dropout_p=0.1,
                 li_sigma=10):
        super().__init__()

        self.n_labels=n_labels
        self.use_norm=use_norm

        self.xlmr = LayerXLMREmbeddings(
            pretrained_path=pretrained_path, hidden_size=hidden_size,
            device=device, use_norm=use_norm,
            use_li=use_li, li_dropout_p=li_dropout_p, li_sigma=li_sigma)
        self.lang_detect=LayerLangDetect(n_labels,hidden_size,dropout_p,head_init_range,device)

    def forward(self, input_ids, labels, labels_mask, valid_mask):
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

        transformer_out=self.xlmr(input_ids)
        logits=self.lang_detect(transformer_out)

        if labels is not None:
            #loss_fct = nn.CrossEntropyLoss()
            loss_fct = nn.NLLLoss()
            # Only keep active parts of the loss
            if labels_mask is not None:
                #active_loss = valid_mask.view(-1) == 1
                #active_logits = logits.view(-1, self.n_labels)[active_loss]
                #active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(logits, labels)
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            return loss
        else:
            return logits

    def encode_word(self, s):
        return self.xlmr.encode_word(s)
