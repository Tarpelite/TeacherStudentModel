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


def split(input_ids_all, input_mask_all, segment_ids_all, label_ids_all, train_size):
    '''
        split train and val data
    '''
    total_size = len(input_ids_all)
    permutation = np.random.choice(total_size, train_size, replace=False)
    input_ids_train = np.array(input_ids_all)[permutation]
    input_ids_val = np.array(input_ids_all)
    input_mask_train = np.array(input_mask_all)[permutation]
    input_mask_val = np.array(input_mask_val)
    segment_ids_train = np.array(segment_ids_all)[permutation]
    segment_ids_val = np.array(segment_ids_all)
    label_ids_train = np.array(label_ids_all)[permutation]
    label_ids_val = np.array(label_ids_all)

    input_ids_val = np.delete(input_ids_val, permutation, axis=0)
    input_mask_val = np.delete(input_mask_val, permutation, axis=0)
    segment_ids_val = np.delete(segment_ids_val, permutation, axis=0)
    label_ids_val = np.delete(label_ids_val, permutation, axis=0)

    train_data = (input_ids_train, input_mask_train, segment_ids_train, label_ids_train)
    val_data = (input_ids_val, input_mask_val, segment_ids_val, label_ids_val)

    return train_data, val_data


def load_train_data(args, input_ids, input_mask, segment_ids, label_ids):
    '''
        make the input data into train set
    '''
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
    '''
        make the input data into evaluate set
    '''
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    eval_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    return eval_dataloader

class InputFeatures(object):
    '''A single set of features of data'''
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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

def sample_data(input_ids, input_mask, segment_ids, probas_val, permutation):
    '''
        Sample validation data with permutation
    '''
    input_ids_stu = np.array(input_ids[permutation])
    input_mask_stu = np.array(input_mask[permutation])
    segment_ids_stu = np.array(segment_ids[permutation])
    label_ids_predict = np.array(probas_val)
    label_ids_stu = np.array(label_ids_predict[permutation])
    label_ids_stu = np.array([np.argmax(x, axis=0) for x in label_ids_stu])
    return input_ids_stu, input_mask_stu, segment_ids_stu, label_ids_stu










