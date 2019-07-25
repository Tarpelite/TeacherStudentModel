import csv
import os

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class OOCLAUSProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        """AUS"""
        return ['Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel', 'Booking-Amendment', 'Booking-FirmUp',
                    'Booking-Request-Attachment', 'CargoRelease', 'CargoRelease-PinRelease', 'CargoRelease-ReleaseInstruction', 'ContainerHandling',
                    'ContainerHandling-EmptyRelease', 'ContainerHandling-Reuse', 'ContainerHandling-EmptyRestitution', 'ContainerHandling-DND',
                    'Customs', 'Documentation-ConsignmentNote-Attachment', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation',
                    'Documentation-BL-Amendment', 'Documentation-BL', 'Documentation-SI-Attachment', 'Documentation-ArrivalNotice',
                    'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment', 'Invoice/Payment-Invoice', 'None', 'Rate', 'Report', 'Unclassified']

    def get_pri_labels(self):
        """See base class."""
        """AUS"""
        return ['Booking', 'CargoRelease', 'ContainerHandling', 'Customs', 'Documentation', 'General', 'Invoice/Payment', 'None', 'Rate',
                    'Report', 'Unclassified']

    def label_to_idx(self, label, label_to_ids):
        idx = [0] * len(label_to_ids)
        for item in label.split(';'):
            idx[label_to_ids[item]] = 1
        return idx

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_to_ids = {k: v for v, k in enumerate(self.get_labels())}
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                # print(line)
                continue
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0].strip()
            text_b = None
            # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
            label = self.label_to_idx(line[1].strip(), label_to_ids)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

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

class TrecProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



class DBpediaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(1, 15)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class YelpProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class AmazonProcessor(DataProcessor):
    """Prcoessor for the Amazon Review data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_labeled.csv")), "train")
    
    def get_unlabeled_examples(self, data_dir):
        """See base class"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_unlabeled.csv")), "unlabeled")
        
    
    def get_dev_examples(self, data_dir):
        """See base class"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "dev")
    
    def get_labels(self):
        """See base class."""
        return ['0', '1']
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1] + line[2]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_tsv(self, input_file):
         with open(input_file, 'r', newline='') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines