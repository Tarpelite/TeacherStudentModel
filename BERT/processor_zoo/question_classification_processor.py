import logging
import os
import tokenization
from processor_zoo.processor import DataProcessor, InputExample


class Qi1Processor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        self.labels = ['numeric', 'description', 'entity', 'human', 'abbreviation', 'location']
        return ['numeric', 'description', 'entity', 'human', 'abbreviation', 'location']
        # self.labels = ['def', 'other', 'speed', 'period', 'instru', 'symbol', 'city', 'dist', 'sport', 'plant', 'perc', 'animal',
        #  'ord', 'temp', 'date', 'letter', 'mount', 'cremat', 'code', 'product', 'veh', 'exp', 'weight', 'country',
        #  'count', 'lang', 'desc', 'state', 'manner', 'currency', 'ind', 'reason', 'termeq', 'religion', 'dismed',
        #  'event', 'food', 'money', 'color', 'word', 'title', 'gr', 'body', 'abb', 'substance', 'volsize', 'techmeth']
        # return ['def', 'other', 'speed', 'period', 'instru', 'symbol', 'city', 'dist', 'sport', 'plant', 'perc', 'animal',
        #  'ord', 'temp', 'date', 'letter', 'mount', 'cremat', 'code', 'product', 'veh', 'exp', 'weight', 'country',
        #  'count', 'lang', 'desc', 'state', 'manner', 'currency', 'ind', 'reason', 'termeq', 'religion', 'dismed',
        #  'event', 'food', 'money', 'color', 'word', 'title', 'gr', 'body', 'abb', 'substance', 'volsize', 'techmeth']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = None
            text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
            label = tokenization.convert_to_unicode(line[2].strip().lower())
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QWProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        # self.labels = ['numeric', 'description', 'entity', 'human', 'abbreviation', 'location']
        return ['1', '0']
        # self.labels = ['def', 'other', 'speed', 'period', 'instru', 'symbol', 'city', 'dist', 'sport', 'plant', 'perc', 'animal',
        #  'ord', 'temp', 'date', 'letter', 'mount', 'cremat', 'code', 'product', 'veh', 'exp', 'weight', 'country',
        #  'count', 'lang', 'desc', 'state', 'manner', 'currency', 'ind', 'reason', 'termeq', 'religion', 'dismed',
        #  'event', 'food', 'money', 'color', 'word', 'title', 'gr', 'body', 'abb', 'substance', 'volsize', 'techmeth']
        # return ['def', 'other', 'speed', 'period', 'instru', 'symbol', 'city', 'dist', 'sport', 'plant', 'perc', 'animal',
        #  'ord', 'temp', 'date', 'letter', 'mount', 'cremat', 'code', 'product', 'veh', 'exp', 'weight', 'country',
        #  'count', 'lang', 'desc', 'state', 'manner', 'currency', 'ind', 'reason', 'termeq', 'religion', 'dismed',
        #  'event', 'food', 'money', 'color', 'word', 'title', 'gr', 'body', 'abb', 'substance', 'volsize', 'techmeth']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = None
            # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
            tem_label = '0'
            if line[1].strip() in ['0.8', '1.0']:
                tem_label = '1'
            label = tokenization.convert_to_unicode(tem_label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
