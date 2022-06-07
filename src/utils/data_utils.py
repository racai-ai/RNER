import torch
from torch.utils.data import TensorDataset
import sys
from utils.NERProcessor import NerProcessor
from utils.LangDetectProcessor import LangDetectProcessor

def create_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids, all_lmask_ids, all_valid_ids)


def get_processor_for_task(task, labels, ner_labels,debug):
    if task == "lang_detect":
        return LangDetectProcessor(labels=labels, ner_labels=ner_labels,debug=debug)
    elif task == "ner" or task=="ner_multi":
        return NerProcessor(labels=ner_labels,debug=debug)
    else:
        print("Unknown task {}".format(task))
        sys.exit(-1)
