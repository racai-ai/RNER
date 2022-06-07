import os
import logging
import sys
import spacy

from utils.data_classes import InputFeatures,InputExample


class NerProcessor:
    """Processor for the CoNLL-2003 data set."""

    def __init__(self, labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"], debug=False):
        self.labels=labels
        self.debug=debug

    def get_train_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_train_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_dev_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "valid.txt"))

    def get_test_examples(self, data_dir, predict=False):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_test_sentences(self, data_dir):
        return self._read_file(os.path.join(data_dir, "test.txt"))

    def get_deploy_examples(self, data_dir, predict=False):
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
                    if self.debug and len(data)>10: break
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

    def convert_examples_to_features(self,examples, label_list, max_seq_length, encode_method, num_seq):
        """Converts a set of examples into XLMR compatible format

        * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
        * Other positions are labeled with 0 ("IGNORE")
        * num_seq is ignored

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
