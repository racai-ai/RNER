import os
import logging
import sys

import torch 
from torch.utils.data import TensorDataset
import spacy

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, valid_ids=None, label_mask=None, token_conllup_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        if token_conllup_ids is None:
            self.token_conllup_ids=[-1 for id in self.input_ids]
        else:
            self.token_conllup_ids = token_conllup_ids


class NerProcessor:
    """Processor for the CoNLL-2003 data set."""

    def __init__(self, labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]):
        self.labels=labels

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_train_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_dev_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "valid.txt"))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_test_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "test.txt"))

    def get_deploy_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "deploy.txt"),ftype='predict'), "deploy")

    def get_deploy_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "deploy.txt"))

    def get_deploy_examples_from_text(self,text,lang):
        data=self._read_text(text,lang)
        return {"examples":self._create_examples(data["data"],"deploy"), "doc":data["doc"]}

    def get_labels(self):
        return self.labels

    def _read_text(self,text,lang):
        data = []
        sentence = []
        label = []

        if lang=="ro":
            nlp = spacy.load("ro_core_news_lg")
        else:
            logging.debug("Unknown language")
            sys.exit(-1)

        doc=nlp(text)
        for token in doc:
            if token.is_sent_start:
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []

            sentence.append(token.text)
            label.append("O")

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return {"data":data, "doc":doc}

    def _read_file(self, filename, ftype='train'):
        '''
        read file
        '''
        f = open(filename)
        data = []
        sentence = []
        label = []

        for i, line in enumerate(f, 1):
            if line.startswith('# id'):
                continue

            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n": # or line[0] == '.'
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue

            splits = line.split()

            if ftype=='predict':
                word=splits[0]
                tag='O'
            else:
                assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
                word, tag = splits[0], splits[-1]
                assert tag in self.get_labels(), "unknown tag {} in line {}".format(tag, i)

            sentence.append(word.strip())
            label.append(tag.strip())

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, encode_method):
    """Converts a set of examples into XLMR compatible format

    * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
    * Other positions are labeled with 0 ("IGNORE")

    """
    ignored_label = "IGNORE"
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    label_map[ignored_label] = 0  # 0 label is to be ignored
    
    features = []
    for (ex_index, example) in enumerate(examples):

        textlist = example.text_a.split(' ')
        labellist = example.label
        labels = []
        valid = []
        label_mask = []
        token_ids = []
       
        for i, word in enumerate(textlist):  
            tokens = encode_method(word.strip())  # word token ids   
            token_ids.extend(tokens)  # all sentence token ids
            label_1 = labellist[i]
            for m in range(len(tokens)):
                if m == 0:  # only label the first BPE token of each work
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    labels.append(ignored_label)  # unlabeled BPE token
                    label_mask.append(0)
                    valid.append(0)

        logging.debug("token ids = ")
        logging.debug(token_ids)
        logging.debug("labels = ")
        logging.debug(labels)
        logging.debug("valid = ")
        logging.debug(valid)

        if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
            token_ids = token_ids[0:(max_seq_length-2)]
            labels = labels[0:(max_seq_length-2)]
            valid = valid[0:(max_seq_length-2)]
            label_mask = label_mask[0:(max_seq_length-2)]

        # adding <s>
        token_ids.insert(0, 0)
        labels.insert(0, ignored_label)
        label_mask.insert(0, 0)
        valid.insert(0, 0)

        # adding </s>
        token_ids.append(2)
        labels.append(ignored_label)
        label_mask.append(0)
        valid.append(0)

        assert len(token_ids) == len(labels)
        assert len(valid) == len(labels)

        label_ids = []
        for i, _ in enumerate(token_ids):
            label_ids.append(label_map[labels[i]])

        assert len(token_ids) == len(label_ids)
        assert len(valid) == len(label_ids)

        input_mask = [1] * len(token_ids)

        while len(token_ids) < max_seq_length:
            token_ids.append(1)  # token padding idx
            input_mask.append(0)
            label_ids.append(label_map[ignored_label])  # label ignore idx
            valid.append(0)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(label_map[ignored_label])
            label_mask.append(0)

        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 2:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                [str(x) for x in token_ids]))
            logging.info("input_ids: %s" %
                         " ".join([str(x) for x in token_ids]))
            logging.info("input_mask: %s" %
                         " ".join([str(x) for x in input_mask]))
            logging.info("label: %s (id = %s)" % (example.label, " ".join(map(str, label_ids))))
            logging.info("label_mask: %s" %
                         " ".join([str(x) for x in label_mask]))
            logging.info("valid mask: %s" %
                        " ".join([str(x) for x in valid]))

        features.append(
            InputFeatures(input_ids=token_ids,
                          input_mask=input_mask,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))

    return features

def makeFeature(ctoken_ids, clabels, clabel_mask, cvalid, max_seq_length, ignored_label, label_map, ctoken_conllup_ids):

    token_ids=ctoken_ids
    labels=clabels
    label_mask=clabel_mask
    valid=cvalid
    token_conllup_ids=ctoken_conllup_ids

    # adding <s>
    token_ids.insert(0, 0)
    token_conllup_ids.insert(0,-1)
    labels.insert(0, ignored_label)
    label_mask.insert(0, 0)
    valid.insert(0, 0)

    # adding </s>
    token_ids.append(2)
    token_conllup_ids.append(-1)
    labels.append(ignored_label)
    label_mask.append(0)
    valid.append(0)

    assert len(token_ids) == len(labels)
    assert len(valid) == len(labels)

    label_ids = []
    for i, _ in enumerate(token_ids):
        label_ids.append(label_map[labels[i]])

    assert len(token_ids) == len(label_ids)
    assert len(valid) == len(label_ids)

    input_mask = [1] * len(token_ids)

    while len(token_ids) < max_seq_length:
        token_ids.append(1)  # token padding idx
        input_mask.append(0)
        label_ids.append(label_map[ignored_label])  # label ignore idx
        valid.append(0)
        label_mask.append(0)
        token_conllup_ids.append(-1)

    while len(label_ids) < max_seq_length:
        label_ids.append(label_map[ignored_label])
        label_mask.append(0)

    assert len(token_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(valid) == max_seq_length
    assert len(label_mask) == max_seq_length
    assert len(token_conllup_ids) == max_seq_length

    return InputFeatures(input_ids=token_ids,
                          input_mask=input_mask,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask,
                          token_conllup_ids=token_conllup_ids)

def create_features_from_conllup(input_file, label_list, max_seq_length, encode_method):
    """Converts a set of examples into XLMR compatible format

    * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
    * Other positions are labeled with 0 ("IGNORE")

    """
    ignored_label = "IGNORE"
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    label_map[ignored_label] = 0  # 0 label is to be ignored
    
    features = []
    conllup = []

    labels = []
    valid = []
    label_mask = []
    token_ids = []
    token_conllup_ids = []
    with open(input_file,"r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line=line.rstrip()
            tokenConllupId=len(conllup)
            conllup.append(line)
            
            if len(line)==0 or line.startswith("#"):
                if len(token_ids)>0:
                    features.append(makeFeature(token_ids,labels,label_mask,valid, max_seq_length, ignored_label, label_map, token_conllup_ids))
                    labels=[]
                    valid=[]
                    label_mask=[]
                    token_ids=[]
                    token_conllup_ids=[]
                continue

            tokenData=line.split("\t")

            word=tokenData[1]
            tokens = encode_method(word.strip())  # word token ids 

            if len(token_ids)+len(tokens)>=max_seq_length-2:
                features.append(makeFeature(token_ids,labels,label_mask,valid, max_seq_length, ignored_label, label_map, token_conllup_ids))
                token_ids=token_ids[-20:]
                labels=labels[-20:]
                label_mask=label_mask[-20:]
                valid=valid[-20:]
                token_conllup_ids=token_conllup_ids[-20:]

            token_ids.extend(tokens)  # all sentence token ids
            for m in range(len(tokens)):
                if m == 0:  # only label the first BPE token of each work
                    labels.append("O")
                    valid.append(1)
                    label_mask.append(1)
                    token_conllup_ids.append(tokenConllupId)
                else:
                    labels.append(ignored_label)  # unlabeled BPE token
                    label_mask.append(0)
                    valid.append(0)
                    token_conllup_ids.append(-1)

            #if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
            #    token_ids = token_ids[0:(max_seq_length-2)]
            #    labels = labels[0:(max_seq_length-2)]
            #    valid = valid[0:(max_seq_length-2)]
            #    label_mask = label_mask[0:(max_seq_length-2)]



    if len(token_ids)>0:
        features.append(makeFeature(token_ids,labels,label_mask,valid, max_seq_length, ignored_label, label_map, token_conllup_ids))

    return {"features":features, "conllup":conllup}


def create_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)
    all_token_conllup_ids = torch.tensor(
        [f.token_conllup_ids for f in features], dtype=torch.long)        

    return TensorDataset(
        all_input_ids, all_label_ids, all_lmask_ids, all_valid_ids,all_token_conllup_ids)