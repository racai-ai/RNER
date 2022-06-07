from model.ModelLangDetect import ModelLangDetect
from model.ModelNER import ModelNER
from model.ModelNERMulti import ModelNERMulti
import sys

def get_model_for_task(
        task,pretrained_path,num_labels,hidden_size,dropout_p,seq_len,device,num_languages,
        use_norm, use_li, li_dropout_p,li_sigma
):
    if task=="lang_detect":
        return ModelLangDetect(
            pretrained_path=pretrained_path,
            n_labels=num_labels, hidden_size=hidden_size,
            dropout_p=dropout_p, device=device,
            use_norm=use_norm, use_li=use_li, li_dropout_p=li_dropout_p, li_sigma=li_sigma
        )
    elif task=="ner":
        # add 1 to num_labels for IGNORE label
        return ModelNER(
            pretrained_path=pretrained_path,
            n_labels=num_labels+1, hidden_size=hidden_size,
            dropout_p=dropout_p, device=device, seq_len=seq_len,
            use_norm=use_norm, use_li=use_li, li_dropout_p=li_dropout_p, li_sigma=li_sigma
        )
    elif task=="ner_multi":
        return ModelNERMulti(pretrained_path=pretrained_path,
                                       n_labels=num_labels+1, hidden_size=hidden_size,
                                       dropout_p=dropout_p, device=device, num_languages=num_languages)
    else:
        print("Unknown task {}".format(task))
        sys.exit(-1)
