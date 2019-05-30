import logging
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
from processor_zoo.processor import DataProcessor, InputExample


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


class OOCLUKDProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ukd_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ukd_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ukd_dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        """AUS"""
        return ['None', 'Report', 'Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel', 'Booking-Amendment',
                'Booking-VGM', 'Booking-Request-Attachment', 'CargoRelease-PinRelease', 'CargoRelease-ReleaseInstruction', 'CargoRelease-PinExtend',
                'ContainerHandling', 'ContainerHandling-BookinRequest', 'ContainerHandling-EmptyRestitution', 'ContainerHandling-DND', 'Customs',
                'Customs-SAD', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation', 'Documentation-BL-Amendment',
                'Documentation-BL', 'Documentation-SI-Attachment', 'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment',
                'Invoice/Payment-Invoice', 'Rate', 'Transportation', 'Transportation-DeliveryNote', 'Transportation-Amendment', 'Transportation-Request',
                'UCR', 'Unclassified']

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


class OOCLAUSTwoProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv.m")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv.m")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "dev")

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
        pri_label_to_ids = {k: v for v, k in enumerate(self.get_pri_labels())}
        for (i, line) in enumerate(lines):
            if len(line) != 3:
                # print(line)
                continue
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0].strip()
            text_b = None
            # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
            label = self.label_to_idx(line[1].strip(), label_to_ids)
            pri_label = self.label_to_idx(line[2].strip(), pri_label_to_ids)
            label = label + pri_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class OOCLAUSUKDProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ukd_train.csv.m")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ukd_dev.csv.m")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "dev")

    def get_labels(self):
        """See base class."""
        """AUS"""
        return ['None', 'Report', 'Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel',
                'Booking-Amendment',
                'Booking-VGM', 'Booking-Request-Attachment', 'CargoRelease-PinRelease',
                'CargoRelease-ReleaseInstruction', 'CargoRelease-PinExtend',
                'ContainerHandling', 'ContainerHandling-BookinRequest', 'ContainerHandling-EmptyRestitution',
                'ContainerHandling-DND', 'Customs',
                'Customs-SAD', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation',
                'Documentation-BL-Amendment',
                'Documentation-BL', 'Documentation-SI-Attachment', 'General-Service/Vessel', 'Invoice/Payment',
                'Invoice/Payment-Payment',
                'Invoice/Payment-Invoice', 'Rate', 'Transportation', 'Transportation-DeliveryNote',
                'Transportation-Amendment', 'Transportation-Request',
                'UCR', 'Unclassified', 'PAD']
        # return ['Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel', 'Booking-Amendment', 'Booking-FirmUp',
        #             'Booking-Request-Attachment', 'CargoRelease', 'CargoRelease-PinRelease', 'CargoRelease-ReleaseInstruction', 'ContainerHandling',
        #             'ContainerHandling-EmptyRelease', 'ContainerHandling-Reuse', 'ContainerHandling-EmptyRestitution', 'ContainerHandling-DND',
        #             'Customs', 'Documentation-ConsignmentNote-Attachment', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation',
        #             'Documentation-BL-Amendment', 'Documentation-BL', 'Documentation-SI-Attachment', 'Documentation-ArrivalNotice',
        #             'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment', 'Invoice/Payment-Invoice', 'None', 'Rate', 'Report', 'Unclassified', 'PAD']

    def get_pri_labels(self):
        """See base class."""
        """AUS"""
        return ['Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel',
                'Booking-Amendment', 'Booking-FirmUp',
                'Booking-Request-Attachment', 'CargoRelease', 'CargoRelease-PinRelease',
                'CargoRelease-ReleaseInstruction', 'ContainerHandling',
                'ContainerHandling-EmptyRelease', 'ContainerHandling-Reuse', 'ContainerHandling-EmptyRestitution',
                'ContainerHandling-DND',
                'Customs', 'Documentation-ConsignmentNote-Attachment', 'Documentation-SI', 'Documentation-BL-Release',
                'Documentation-BL-Confirmation',
                'Documentation-BL-Amendment', 'Documentation-BL', 'Documentation-SI-Attachment',
                'Documentation-ArrivalNotice',
                'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment', 'Invoice/Payment-Invoice',
                'None', 'Rate', 'Report', 'Unclassified', 'PAD']
        # return ['None', 'Report', 'Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel',
        #         'Booking-Amendment',
        #         'Booking-VGM', 'Booking-Request-Attachment', 'CargoRelease-PinRelease',
        #         'CargoRelease-ReleaseInstruction', 'CargoRelease-PinExtend',
        #         'ContainerHandling', 'ContainerHandling-BookinRequest', 'ContainerHandling-EmptyRestitution',
        #         'ContainerHandling-DND', 'Customs',
        #         'Customs-SAD', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation',
        #         'Documentation-BL-Amendment',
        #         'Documentation-BL', 'Documentation-SI-Attachment', 'General-Service/Vessel', 'Invoice/Payment',
        #         'Invoice/Payment-Payment',
        #         'Invoice/Payment-Invoice', 'Rate', 'Transportation', 'Transportation-DeliveryNote',
        #         'Transportation-Amendment', 'Transportation-Request',
        #         'UCR', 'Unclassified', 'PAD']

    def label_to_idx(self, label, label_to_ids):
        idx = [0] * len(label_to_ids)
        for item in label.split(';'):
            idx[label_to_ids[item]] = 1
        return idx

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_to_ids = {k: v for v, k in enumerate(self.get_labels())}
        pri_label_to_ids = {k: v for v, k in enumerate(self.get_pri_labels())}
        for (i, line) in enumerate(lines):
            if len(line) != 3:
                # print(line)
                continue
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0].strip()
            text_b = None
            # text_b = tokenization.convert_to_unicode(' '.join(self.labels).lower())
            aus_label = self.label_to_idx(line[1].strip(), label_to_ids)
            ukd_label = self.label_to_idx(line[2].strip(), pri_label_to_ids)
            label = aus_label + ukd_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples