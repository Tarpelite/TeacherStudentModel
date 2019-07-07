# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time
import collections
import numpy as np
import torch
import copy
import requests

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling_for_doc import BertForDocMultiClassification
from oocl_utils.evaluate import evaluation_report
from teacher_student_model.processor_zoo import OOCLAUSProcessor
from teacher_student_model.fct_utils import train
from oocl_utils.score_output_2_labels import convert
from sklearn.neighbors import KNeighborsClassifier



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

global_step = 0
nb_tr_steps = 0
tr_loss = 0


def accuracy(out, labels):

    return

def vote_for_choice(acc_t, acc_s1, acc_s2):
    '''
        vote for the best model
    '''
    acc_list = [acc_t, acc_s1, acc_s2]
    return acc_list.index(max(acc_list))


def top_k_choose(probas_val, top_k):
    '''
        choose the top_k data from the predict data
        params:probas_val:logits
               top_k: the num of top k elenments
        return:permutation of the data
    '''
    label_ids_predict = np.array(probas_val)
    bitmap = np.zeros(len(probas_val))
    class_num = label_ids_predict.shape[1]
    permutation = []
    for i in range(class_num):
        pos_sort = sorted(range(len(probas_val)), key=lambda k: probas_val[k][i], reverse=True)
        pos_sort = pos_sort[:top_k]
        for pos in pos_sort:
            bitmap[pos] = 1   
    for i in range(len(bitmap)):
        if bitmap[i] == 1:
            permutation.append(i)
    permutation = np.array(permutation)
    return permutation

def balance_top_k_choose(probas_val, top_k):
    '''
        the distribution of classes is balanced and reach the num of top k.
        try best to balance.
    '''
    probas_val = np.array(probas_val)
    labels = np.argmax(probas_val, axis=1)
    logger.info("label distribution = %s", collections.Counter(labels))
    classes = len(probas_val[0])
    pos = 0
    classes_unsort = [[]for x in range(classes)]
    permutation = []

    for label in labels:
        row = [pos, probas_val[pos][label]]
        classes_unsort[label].append(row)
        pos += 1
    # check min_len >= k
    # min_len = np.min([len(x) for x in classes_unsort])
    # assert(min_len >= top_k)
    for i in range(classes):
        class_i = classes_unsort[i]
        class_i.sort(key = lambda k: k[1], reverse=True)
        class_len = len(class_i)
        if class_len < top_k:
            for row in class_i:
                permutation.append(row[0])
        else:
            for row in class_i[:top_k]:
                permutation.append(row[0])
    
    permutation = np.array(permutation)
    #print("permutation", permutation)
    return permutation
            

def evaluate_model(model, device, eval_data_loader, logger):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    idx = 0
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader, desc="Iteration"):
        idx += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        outputs = np.argmax(logits, axis=1)

        if idx < 4:
            logger.info('  prediction label = %s, reference label = %s', outputs, label_ids)

        tmp_eval_accuracy = np.sum(outputs == label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_examples

    logger.info("accuracy: %f ", eval_accuracy)
    return eval_accuracy

def create_model(args, cache_dir, num_labels, device):
    '''
        create new model
    '''
    model_new = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                            cache_dir = cache_dir,
                                                            num_labels=num_labels)
    if args.fp16:
        model_new.half()    
    model_new.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model_new = DDP(model_new)
    elif torch.cuda.device_count() > 1:
         model_new = torch.nn.DataParallel(model_new)
    return model_new
    

def predict_model(model, args, eval_dataloader, device):
    '''
        predict val data
    '''
    predict_result = []
    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        predict_result.extend(logits)
    return predict_result


def split(input_ids_all, input_mask_all, segment_ids_all, label_ids_all, train_size):
    '''
        split train and val data
    '''
    total_size = len(input_ids_all)
    # print("total_size", total_size)
    # print("train_size", train_size)
    permutation = np.random.choice(total_size, train_size, replace=False)
    input_ids_train = np.array(input_ids_all)[permutation]
    input_ids_val = np.array(input_ids_all)[permutation]
    input_mask_train = np.array(input_mask_all[permutation])
    input_mask_val = np.array(input_mask_all[permutation])
    segment_ids_train = np.array(segment_ids_all[permutation])
    segment_ids_val = np.array(segment_ids_all)[permutation]
    label_ids_train = np.array(label_ids_all)[permutation]
    label_ids_val = np.array(label_ids_all)[permutation]
    train_data = (input_ids_train, input_mask_train, segment_ids_train, label_ids_train)
    val_data = (input_ids_val, input_mask_val, segment_ids_val, label_ids_val)

    return train_data, val_data


def load_train_data(args, input_ids, input_mask, segment_ids, label_ids):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    train_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader


def load_eval_data(args, input_ids, input_mask, segment_ids, label_ids):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    eval_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return eval_dataloader


def get_k_random_samples(input_ids, input_mask, segment_ids, label_ids, initial_labeled_samples, trainset_size):
    permutation = np.random.choice(trainset_size, initial_labeled_samples, replace=False)
    print('initial random chosen samples', permutation.shape)

    input_ids_train = input_ids[permutation]
    input_mask_train = input_mask[permutation]
    segment_ids_train = segment_ids[permutation]
    label_ids_train = label_ids[permutation]

    return permutation, input_ids_train, input_mask_train, segment_ids_train, label_ids_train


def init_student_weights(model_TL, model_TU, model_student, alpha):
    '''
        W_s = alpha*W_tl + (1-alpha)*W_tu
    '''
    TL_dict = model_TL.state_dict()
    TU_dict = model_TU.state_dict()
    S_dict = model_student.state_dict()
    res_dict= copy.deepcopy(S_dict)

    for item in TL_dict:
        res_dict[item]  = alpha*TL_dict[item] + (1-alpha)*TU_dict[item]
    
    model_student.load_state_dict(res_dict)
    return model_student


class BaseSelectionFunction(object):

    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)

        return selection


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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))
        '''
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def sample_data(args, processor, label_list, tokenizer, permutation, probas_val):
    '''
        Sample valiadation data with permutation
    '''

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    input_ids_stu = np.array(all_input_ids[permutation])
    input_mask_stu = np.array(all_input_mask[permutation])
    segment_ids_stu = np.array(all_segment_ids[permutation])
    label_ids_predict = np.array(probas_val)
    label_ids_stu = np.array(label_ids_predict[permutation])
    label_ids_stu = np.array([np.argmax(x, axis=0) for x in label_ids_stu])
    return input_ids_stu, input_mask_stu, segment_ids_stu, label_ids_stu

def write_result(args, result:list):
    '''
        wirte the accuracy into the file
        result Example:

            format like:
            teacher model, student model 1, student model 2
            Iter1: 0.632339,0.729580,0.739026
            Iter2: 0.739026,0.724208,0.778477

            data like:
            [
                [0.632339, 0.729580, 0.739026],
                [0.739026, 0.724208, 0.778477]
            ]
    '''
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    path = os.path.join(args.output_dir, "acc_result.txt")
   
    with open(path , "w+", encoding="utf-8") as f:
        table_head = "\tTeacherModel" + "\tStudentModel1" + "\tStudentModel2" + "\n"
        f.write(table_head)
        for row in result:
            line = ""
            for data in row:
                line += str(data)  + "\t"
            line += "\n"
            f.write(line)

def cook_data(args, processor, label_list, tokenizer):
    '''
        Cook training and dev data for teacher model
    '''
    logger.info("***** Cook training and dev data for teacher model *****")
    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)

    input_ids_train = np.array([f.input_ids for f in train_features])
    input_mask_train = np.array([f.input_mask for f in train_features])
    segment_ids_train = np.array([f.segment_ids for f in train_features])
    label_ids_train = np.array([f.label_id for f in train_features])
    train_data_loader = load_train_data(args, input_ids_train, input_mask_train, segment_ids_train, label_ids_train)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return train_data_loader, eval_data_loader


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_balance",
                        action="store_true",
                        help="Set this flag if you want to use the balanced choose function")
    parser.add_argument("--push_message",
                        action = "store_true",
                        help="set this flag if you want to push message to your phone")
    parser.add_argument("--top_k",
                        default=500,
                        type=int,
                        help="Set the num of top k pseudo labels Teacher will choose for Student to learn")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--num_student_train_epochs",
                        type=float,
                        default=3.0,
                        help="Total number of student model training epochs to perform.")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.05,
                        help="threshold for improvenesss of model")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.33,
                        help = "the weights of the TL model in the final model")
    parser.add_argument("--ft_true",
                        action="store_true",
                        help="fine tune the student model with true data")
    parser.add_argument("--ft_pseudo",
                        action="store_true",
                        help="fine tune the student model with pseudo data")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "aus": OOCLAUSProcessor,
        "dbpedia": DBpediaProcessor,
        "trec": TrecProcessor,
        "yelp": YelpProcessor,
    }

    num_labels_task = {
        "aus": 33,
        "dbpedia": len(DBpediaProcessor().get_labels()),
        "trec": len(TrecProcessor().get_labels()),
        "yelp": len(YelpProcessor().get_labels()),
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    logger.info("***** Build teacher(label) model *****")
    # Prepare model
    cache_dir = args.cache_dir
    model_TL  = create_model(args, cache_dir, num_labels, device)

    logger.info("***** Build teacher(unlabel) model *****")
    cache_dir = args.cache_dir
    model_TU  = create_model(args, cache_dir, num_labels, device)
    logger.info("***** Build student model *****")

    model_student = create_model(args, cache_dir, num_labels, device)
    
    logger.info("***** Finish TL, TU and Student model building *****")

    if args.do_train:
        # step 0: load train examples
        train_data_loader, eval_data_loader = cook_data(args, processor, label_list, tokenizer)
        
        # step 1: train the TL model with labeled data
        logger.info("***** Running train TL model with labeled data *****")
        model_TL = train(model_TL, args, n_gpu, train_data_loader, device, args.num_train_epochs, logger)
        model_TL.to(device)

        logger.info("***** Evaluate TL model *****")
        model_TL_accuracy = evaluate_model(model_TL, device, eval_data_loader, logger)

        # Step 2: predict the val_set
        logger.info("***** Product pseudo label from TL model *****")
        probas_val = predict_model(model_TL, args, eval_data_loader, device)
    

        # Step 3: choose top-k data_val and reset train_data
        if args.do_balance:
            permutation = balance_top_k_choose(probas_val, args.top_k)
        else:
            permutation = top_k_choose(probas_val, args.top_k)
        input_ids_TU, input_mask_TU, segment_ids_TU, label_ids_TU = sample_data(args, processor, label_list, tokenizer, permutation, probas_val)
        logger.info("Pseudo label distribution = %s", collections.Counter(label_ids_TU))

        # step 4: train TU model with Pseudo labels
        logger.info("***** Running train TU model with pseudo data *****")
        train_data_loader_TU = load_train_data(args, input_ids_TU, input_mask_TU, segment_ids_TU, label_ids_TU)
    
        model_TU = train(model_TU, args, n_gpu, train_data_loader_TU, device, args.num_train_epochs, logger)
        model_TU.to(device)

        logger.info("***** Evaluate TU model  *****")
        model_TU_accuracy = evaluate_model(model_TU, device, eval_data_loader, logger)

        # step 5: init student model with mix weights from TL and TU model
        logger.info("***** Init student model with weights from TL and TU model *****")
        model_student = init_student_weights(model_TL, model_TU, model_student, args.alpha)
        model_student.to(device)

        if args.ft_true:
            # step 6: train student model with train data
            logger.info("***** Running train student model with train data *****")
            model_student = train(model_student, args, n_gpu, train_data_loader, device, args.num_student_train_epochs, logger)
            model_student.to(device)

        if args.ft_pseudo:
            # step 7: train student model with Pseudo labels
            logger.info("***** Running train student model with Pseudo data *****")
            model_student = train(model_student, args, n_gpu, train_data_loader_TU, device, args.num_student_train_epochs, logger)
            model_student.to(device)
        
        logger.info("***** Evaluate student model  *****")
        model_student_accuracy = evaluate_model(model_student, device, eval_data_loader, logger)
    

        results = [model_TL_accuracy, model_TU_accuracy, model_student_accuracy]
        print(results)

        if args.push_message:
            api = "https://sc.ftqq.com/SCU47715T1085ec82936ebfe2723aaa3095bb53505ca315d2865a0.send"
            title = args.task_name
            if args.ft_true:
                title += " ft_true "
            if args.ft_pseudo:
                title += "ft_pseudo"
            content = ""
            content += "Params: alpha:{} \n".format(str(args.alpha)) 
            content += "model_TL: " + str(model_TL_accuracy) + "\n"
            content += "model_TU: " + str(model_TU_accuracy) + "\n"
            content += "model_student: " + str(model_student_accuracy) + "\n"
            data = {
                "text":title,
                "desp":content
            }
            requests.post(api, data=data)
        
if __name__ == "__main__":
    main()
