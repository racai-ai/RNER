from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LateralInhibition import LateralInhibition
from model.LayerXLMREmbeddings import LayerXLMREmbeddings
from model.LayerLangDetect import LayerLangDetect

class ModelNERMulti(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p, num_languages, label_ignore_idx=0,
                head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels
        self.num_languages = num_languages
        
        self.li = LateralInhibition(hidden_size, [0.5])
        self.li_dropout = nn.Dropout(dropout_p)

        self.lang_linear_0=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_0=nn.Dropout(dropout_p)
        self.lang_class_0=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_1=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_1=nn.Dropout(dropout_p)
        self.lang_class_1=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_2=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_2=nn.Dropout(dropout_p)
        self.lang_class_2=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_3=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_3=nn.Dropout(dropout_p)
        self.lang_class_3=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_4=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_4=nn.Dropout(dropout_p)
        self.lang_class_4=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_5=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_5=nn.Dropout(dropout_p)
        self.lang_class_5=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_6=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_6=nn.Dropout(dropout_p)
        self.lang_class_6=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_7=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_7=nn.Dropout(dropout_p)
        self.lang_class_7=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_8=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_8=nn.Dropout(dropout_p)
        self.lang_class_8=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_9=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_9=nn.Dropout(dropout_p)
        self.lang_class_9=nn.Linear(hidden_size, n_labels).to(device)

        self.lang_linear_10=nn.Linear(hidden_size,hidden_size).to(device)
        self.lang_dropout_10=nn.Dropout(dropout_p)
        self.lang_class_10=nn.Linear(hidden_size, n_labels).to(device)


        
        self.label_ignore_idx = label_ignore_idx

        self.xlmr=LayerXLMREmbeddings(pretrained_path,hidden_size,dropout_p,device)
        self.lang_detect=LayerLangDetect(self.num_languages,hidden_size,dropout_p,head_init_range,device)

        self.device=device

        # initializing classification head
        #self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)

    def soft_argmax(self,data):
        alpha = 1000.0 
        N,C = data.shape
        sm=nn.functional.softmax(data*alpha, dim=1)
        r=torch.arange(start=0,end=C).unsqueeze(0).to(self.device)
        return torch.sum(sm * r, dim=1)
        # tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

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
        transformer_out=self.xlmr(inputs_ids)
        logits_ld=self.lang_detect(transformer_out)

        #print("inputs_ids={}".format(inputs_ids.size()))

        lang_id=self.soft_argmax(logits_ld)

        #print("lang_id={}".format(lang_id.size()))

        c1=8

        lang_logits=[]
        
        i=0
        t=F.relu(self.lang_linear_0(transformer_out))
        t=self.lang_dropout_0(t)
        t=self.lang_class_0(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=1
        t=F.relu(self.lang_linear_1(transformer_out))
        t=self.lang_dropout_1(t)
        t=self.lang_class_1(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=2
        t=F.relu(self.lang_linear_2(transformer_out))
        t=self.lang_dropout_2(t)
        t=self.lang_class_2(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=3
        t=F.relu(self.lang_linear_3(transformer_out))
        t=self.lang_dropout_3(t)
        t=self.lang_class_3(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=4
        t=F.relu(self.lang_linear_4(transformer_out))
        t=self.lang_dropout_4(t)
        t=self.lang_class_4(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=5
        t=F.relu(self.lang_linear_5(transformer_out))
        t=self.lang_dropout_5(t)
        t=self.lang_class_5(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=6
        t=F.relu(self.lang_linear_6(transformer_out))
        t=self.lang_dropout_6(t)
        t=self.lang_class_6(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=7
        t=F.relu(self.lang_linear_7(transformer_out))
        t=self.lang_dropout_7(t)
        t=self.lang_class_7(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=8
        t=F.relu(self.lang_linear_8(transformer_out))
        t=self.lang_dropout_8(t)
        t=self.lang_class_8(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=9
        t=F.relu(self.lang_linear_9(transformer_out))
        t=self.lang_dropout_9(t)
        t=self.lang_class_9(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        i=10
        t=F.relu(self.lang_linear_10(transformer_out))
        t=self.lang_dropout_10(t)
        t=self.lang_class_10(t)
        t=torch.mul(t.permute(1,2,0),torch.sigmoid((lang_id-(i-0.5))*c1) * torch.sigmoid(((i+0.5)-lang_id)*c1)).permute(2,0,1)
        lang_logits.append(t)

        stacked=torch.stack(lang_logits)
        #print("stacked={}".format(stacked.size()))

        logits=torch.sum(stacked,0)

        #print("logits={}".format(logits.size()))

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
        return self.xlmr.encode_word(s)
