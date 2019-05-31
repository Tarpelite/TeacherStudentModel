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

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling_for_doc import BertForDocMultiClassification
from processor_zoo.oocl_processor import OOCLAUSProcessor

from oocl_utils.evaluate import evaluation_report
from oocl_utils.score_output_2_labels import convert

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    print(outputs)
    return np.sum(outputs == labels)


def evaluate(model, args, processor, device, global_step, task_name,label_list,tokenizer,report_path):

    #global global_step
    eval_examples = processor.get_dev_examples(args.data_dir)


    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    


    model.eval()
    output_file_path = os.path.join(args.output_dir, "results_epoch_{0}.txt".format(10))
    output_file_writer = open(output_file_path, 'w', encoding='utf-8')
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        #tmp_eval_accuracy = accuracy(logits, label_ids)
        #print(logits)

        for logit, label_id in zip(logits, label_ids):
            output_file_writer.write(str(logit).replace('\n', ' '))
            output_file_writer.write('\t')
            output_file_writer.write(str(label_id).replace('\n', ' '))
            output_file_writer.write('\n')

        # eval_loss += tmp_eval_loss.mean().item()
        #eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    #eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
                'global_step': global_step,
                # 'loss': tr_loss / nb_tr_steps
                }

    output_eval_file = os.path.join(args.output_dir, "eval_results_epoch_{0}.txt".format(10))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    # Save pytorch-model
    output_model_file = os.path.join(args.output_dir, "fine_tune_model_epoch_{0}.bin".format(10))
    torch.save(model.state_dict(), output_model_file)

    aus_label_path = args.data_dir + r"/aus_dev.label.csv"
    ukd_label_path = args.data_dir + r"/ukd_dev.label.csv"
    if task_name == 'aus':
        label_path = aus_label_path
    else:
        label_path = ukd_label_path
    output_file_writer.close()

    convert(output_file_path, task_name)
    evaluation_report(label_path, output_file_path + '.label', task_name.upper(), output_file_path + '.eval', report_path)
    
    true_file = label_path
    pred_file = output_file_path + ".label"

    true_labels = open(true_file, 'r', encoding='utf-8').readlines()
    true_labels = [true_label.strip() for true_label in true_labels]
    pred_labels = open(pred_file, 'r', encoding='utf-8').readlines()
    pred_labels = [pred_label.strip() for pred_label in pred_labels]
    print("true labels", true_labels)
    print("pred labels", pred_labels)


def predict(model, args, eval_dataloader, device):
    '''
        predict val data
    '''
    predict_result = []
    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        
        logits = logits.detach().cpu().numpy()
        predict_result.extend(logits)
    
    predict_result = np.array(predict_result)
    return predict_result

def split(input_ids_all, input_mask_all, segment_ids_all, label_ids_all, train_size):
    '''
        split train and val data
    '''
    total_size = len(input_ids_all)
    #print("total_size", total_size)
    #print("train_size", train_size)
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

    input_ids = torch.tensor(input_ids, dtype = torch.long)
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

    input_ids = torch.tensor(input_ids, dtype = torch.long)
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


def train(model, args, n_gpu, optimizer, num_train_optimization_steps, num_labels,  train_dataloader, device):
    '''
        train model
    '''
    model.train()
    global global_step
    global nb_tr_steps
    global tr_loss
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    ## save model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    config = BertConfig(output_config_file)
    model = BertForDocMultiClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(output_model_file))

    return model




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
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

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

        label_id = example.label
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
        "aus" : OOCLAUSProcessor,
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
    }

    num_labels_task = {
        "aus": 33,
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
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

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir 
    model_teacher = BertForDocMultiClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels = num_labels)
    model_student = BertForDocMultiClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels = num_labels)
    if args.fp16:
        model_teacher.half()
        model_student.half()
    model_teacher.to(device)
    model_student.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model_teacher = DDP(model_teacher)
        model_student = DDP(model_student)
    elif n_gpu > 1:
        model_teacher = torch.nn.DataParallel(model_teacher)
        model_student = torch.nn.DataParallel(model_student)

    # Prepare optimizer
    param_optimizer = list(model_teacher.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps) 

        all_input_ids = np.array([f.input_ids for f in train_features])
        all_input_mask = np.array([f.input_mask for f in train_features])
        all_segment_ids = np.array([f.segment_ids for f in train_features])
        all_label_ids = np.array([f.label_id for f in train_features])
        

        initial_labeled_samples = 5000
        trainset_size = 10000

        # step -2: randomly make the trainset
        train_data, val_data = split(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, trainset_size)
        (input_ids, input_mask, segment_ids, label_ids) = train_data

        # step -1: randomly setup the initial_labeld_samples
        permutation, input_ids_train, input_mask_train, segment_ids_train, label_ids_train = \
            get_k_random_samples(input_ids, input_mask, segment_ids, label_ids, initial_labeled_samples, trainset_size)
        
        # step 0: assgin the valset the rest of the "unlabelled" traning data
        input_ids_val = np.copy(input_ids)
        input_mask_val = np.copy(input_mask)
        segment_ids_val = np.copy(segment_ids)
        label_ids_val = np.copy(label_ids)
        input_ids_val = np.delete(input_ids_val, permutation, axis=0)
        input_mask_val = np.delete(input_mask_val, permutation, axis=0)
        segment_ids_val = np.delete(segment_ids_val, permutation, axis=0)
        label_ids_val = np.delete(label_ids_val, permutation, axis=0)
        print('val set:',input_ids_val.shape, label_ids_val.shape, permutation.shape)

        # Step 1: train the teacher_model
        print("*"*10+"train teacher model"+"*"*10)
        train_dataloader = load_train_data(args, input_ids_train, input_mask_train, segment_ids_train, label_ids_train)
        model_teacher = train(model_teacher, args, n_gpu, optimizer, num_train_optimization_steps, num_labels, train_dataloader, device)
        model_teacher.to(device)

        # Step 2: predict the val_set
        eval_dataloader = load_eval_data(args, input_ids_val, input_mask_val, segment_ids_val, label_ids_val)
        probas_val = predict(model_teacher, args, eval_dataloader, device)

        threshold = 0.5
        label_ids_predict = []
        for i in range(len(probas_val)):
            prob_row = list(probas_val[i])
            label_predict_row = [0 for x in range(probas_val.shape[1])]
            if max(prob_row) > threshold:
                for j, score in enumerate(prob_row):
                    if score > threshold:
                        label_predict_row[j] = 1
            else:
                label_predict_row[prob_row.index(max(prob_row))] = 1
            label_ids_predict.append(label_predict_row)

        label_ids_predict = np.array(label_ids_predict) 

        # Step 3: choose top-k data_val and reset train_data
        pos_list = [0 for x in range(len(probas_val))]
        top_k = 200
        type_len = probas_val.shape[1]
        index_list = []
        for i in range(type_len):
            pos_sort = sorted(range(len(probas_val)), key=lambda k:probas_val[k][i])
            pos_sort = pos_sort[:top_k]
            for pos in pos_sort:
                pos_list[pos] = 1
        for i in range(len(pos_list)):
            if pos_list[i] == 1:
                index_list.append(i)
        permutation = np.array(index_list)

        input_ids_stu = input_ids_val[permutation]
        input_mask_stu = input_ids_val[permutation]
        segment_ids_stu = segment_ids_val[permutation]
        label_ids_stu = label_ids_predict[permutation]
        
        # step 4: train student model with teacher labeled data
        print("*"*10+"train student model"+"*"*10)
        print("train set:", input_ids_stu.shape)
        train_dataloader_stu = load_train_data(args, input_ids_stu, input_mask_stu, segment_ids_stu, label_ids_stu)
        model_student = train(model_teacher, args, n_gpu, optimizer, num_train_optimization_steps, num_labels, train_dataloader_stu, device)
        model_student.to(device)

        # step 5: train student model with true data
        print("*"*10+"refine student model"+"*"*10)
        print("train set:", input_ids_train.shape)
        model_student = train(model_teacher, args, n_gpu, optimizer, num_train_optimization_steps, num_labels, train_dataloader, device)
        model_student.to(device)

    # do eval
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("***** Running  teacher evaluation *****")
        report_path = os.path.join(args.output_dir, "final_report_teacher.csv")
        evaluate(model_student, args, processor, device, global_step, task_name, label_list, tokenizer, report_path)
       
        logger.info("***** Running  student evaluation *****")
        report_path = os.path.join(args.output_dir, "final_report_student.csv")
        evaluate(model_student, args, processor, device, global_step, task_name, label_list, tokenizer, report_path)

        



if __name__ == "__main__":
    main()
