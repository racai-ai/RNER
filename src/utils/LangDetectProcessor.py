import os
import logging
from utils.NERProcessor import NerProcessor
import sys
from utils.data_classes import InputFeatures

class LangDetectProcessor:
    """Processor for language detection."""

    def __init__(self, labels=["en", "ro"], ner_labels=[], debug=False):
        self.labels=labels
        self.ner_labels=ner_labels
        self.debug=debug

    def get_train_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(data_dir,"train", predict)

    def get_train_sentences(self, data_dir):
        return []

    def get_dev_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(data_dir,"valid", predict)

    def get_dev_sentences(self, data_dir):
        return []

    def get_test_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(data_dir,"test", predict)

    def get_test_sentences(self, data_dir):
        return []

    def get_deploy_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(data_dir,"deploy", predict)

    def get_deploy_sentences(self, data_dir):
        return []

    def get_labels(self):
        return self.labels

    def _create_examples(self, data_dir, set_type, predict):

        all_examples=[]

        paths=[]
        if not predict:
            for l in self.labels:
                path=os.path.join(data_dir, l)
                paths.append((path,l))
        else:
            paths.append((data_dir,self.labels[0]))

        for (path,l) in paths:
            np=NerProcessor(self.ner_labels,self.debug)
            if set_type=="train":
                examples=np.get_train_examples(path)
            elif set_type=="valid":
                examples=np.get_dev_examples(path)
            elif set_type=="test":
                examples=np.get_test_examples(path)
            elif set_type=="deploy":
                examples=np.get_deploy_examples(path)
            else:
                print("LangDetectProcessor -> _create_examples -> unknown set type [{}]".format(set_type))
                sys.exit(-1)

            for ex in examples:
                ex.label=l
                all_examples.append(ex)

        return all_examples

    def convert_examples_to_features(self,examples, label_list, max_seq_length, encode_method, num_seq):
        """Converts a set of examples into XLMR compatible format

        * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
        * Other positions are labeled with 0 ("IGNORE")

        """
        label_map = {label: i for i, label in enumerate(label_list, 0)}

        features = []
        for (ex_index, example) in enumerate(examples):

            textlist = example.text_a.split(' ')
            valid = []
            token_ids = []

            for i, word in enumerate(textlist):
                tokens = encode_method(word.strip())  # word token ids
                token_ids.extend(tokens)  # all sentence token ids
                for m in range(len(tokens)):
                    if m == 0:  # only label the first BPE token of each work
                        valid.append(1)
                    else:
                        valid.append(0)

            logging.debug("token ids = ")
            logging.debug(token_ids)
            logging.debug("valid = ")
            logging.debug(valid)

            if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
                #token_ids = token_ids[0:(max_seq_length-2)]
                #valid = valid[0:(max_seq_length-2)]
                token_ids1=token_ids[0:128]
                valid1=valid[0:128]
                remain=max_seq_length-2-128
                if remain>0:
                    token_ids1=token_ids1+token_ids[-remain:]
                    valid1=valid1+valid[-remain:]
                if len(token_ids1)>=max_seq_length-1:
                    token_ids1=token_ids1[0:(max_seq_length-2)]
                    valid1=valid1[0:(max_seq_length-2)]
                token_ids=token_ids1
                valid=valid1

            # adding <s>
            token_ids.insert(0, 0)
            valid.insert(0, 0)

            # adding </s>
            token_ids.append(2)
            valid.append(0)

            assert len(token_ids) == len(valid)

            input_mask = [1] * len(token_ids)

            while len(token_ids) < max_seq_length:
                token_ids.append(1)  # token padding idx
                input_mask.append(0)
                valid.append(0)

            assert len(token_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(valid) == max_seq_length

            if ex_index < 2:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % " ".join(
                    [str(x) for x in token_ids]))
                logging.info("input_ids: %s" %
                             " ".join([str(x) for x in token_ids]))
                logging.info("input_mask: %s" %
                             " ".join([str(x) for x in input_mask]))
                logging.info("valid mask: %s" %
                            " ".join([str(x) for x in valid]))

            features.append(
                InputFeatures(input_ids=token_ids,
                              input_mask=input_mask,
                              label_id=label_map[example.label],
                              valid_ids=valid,
                              label_mask=[]))

        return features
