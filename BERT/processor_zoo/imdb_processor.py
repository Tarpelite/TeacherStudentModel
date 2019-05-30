import logging
import os
import tokenization
from processor_zoo.processor import DataProcessor, InputExample
from nltk.tokenize.punkt import PunktSentenceTokenizer


sentence_tokenizer = PunktSentenceTokenizer()


class IMDBProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mini_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mini_train.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                # print(line)
                continue
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1].strip())
            text_b = None
            # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
            label = tokenization.convert_to_unicode(line[0].strip().lower())
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IMDBProcessorV2(DataProcessor):
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
        return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        max_sentence = 40
        padding = ''
        examples = []
        i = -1
        for line in lines:
            if len(line) != 2:
                # print(line)
                continue
            guid = "%s-%s" % (set_type, i)
            sentences = sentence_tokenizer.tokenize(line[1].strip())
            for sentence in sentences[:min(len(sentences), max_sentence)]:
                i += 1
                text_a = tokenization.convert_to_unicode(sentence)
                text_b = None
                # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
                label = tokenization.convert_to_unicode(line[0].strip().lower())
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            if len(sentences) < max_sentence:
                for j in range(max_sentence - len(sentences)):
                    text_a = tokenization.convert_to_unicode(padding)
                    text_b = None
                    # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
                    label = tokenization.convert_to_unicode(line[0].strip().lower())
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples