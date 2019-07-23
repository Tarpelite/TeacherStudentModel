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
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling_for_doc import BertForDocMultiClassification
from oocl_utils.evaluate import evaluation_report
from teacher_student_model.processor_zoo import OOCLAUSProcessor, DBpediaProcessor, TrecProcessor, YelpProcessor
from teacher_student_model.fct_utils import train, init_optimizer, create_model, predict_model, init_student_weights
from oocl_utils.score_output_2_labels import convert
from sklearn.neighbors import KNeighborsClassifier
from teacher_student_model.selection_zoo import RandomSelectionFunction, TopkSelectionFunction, BalanceTopkSelectionFunction
from teacher_student_model.IO_utils import split, load_train_data, load_eval_data, convert_examples_to_features, sample_data


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskTriModel(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MultiTaskTriModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask=None, labels=None, mode=0):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # sum_repre = torch.bmm(torch.unsqueeze(attention_mask, 1).float(), sequence_output).squeeze()
        # pooled_output = torch.div(sum_repre.t(), torch.sum(attention_mask, -1).float()).t()
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if mode == 0:
            # norm mode: using the default classifier
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
            else:
                return logits
        elif mode == 1:
            # pooled mode: using the another classifer
            return pooled_output


def init_tri_model(model, args, n_gpu, train_dataloader, device, num_epoch, logger):
    '''
        the initial process of training
    '''
    logger.info("***** Initial optimizer *****")
    optimizer, num_train_optimization_steps = init_optimizer(model, args, train_dataloader)

    logger.info("***** Running train *****")
    model.train()
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    for _ in trange(int(num_epoch), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, mode=0)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
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
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
    return model



def create_tri_model(args, cache_dir, num_labels, device):
    '''
        create tri model
    '''
    model_new = MultiTaskTriModel.from_pretrained(args.bert_model,
                                                 cache_dir = cache_dir,
                                                 num_labels = num_labels)
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

def train_model(model, classifier2, classifier3, args, n_gpu, train_dataloader, device, num_epoch, logger, turn):
    
    # init 3 different optimizer
    optimizer, num_train_optimization_steps = init_optimizer(model, args, train_dataloader)
    classifier1 = model.classifier
    model.classifier = classifier2
    optimizer2, _ = init_optimizer(model, args, train_dataloader)
    model.classifier = classifier3
    optimizer3, _ = init_optimizer(model, args, train_dataloader)
    model.classifier = classifier1

    logger.info("***** Running train {} *****".format(turn))
    model.train()
    classifier2.train()
    classifier3.train()
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    gamma = 0.01
    # print("Before training:", )

    for _ in trange(int(num_epoch), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            pooled_output = model(input_ids, segment_ids, input_mask, label_ids, mode=1)
            logits1 = model.classifier(pooled_output)
            logits2 = classifier2(pooled_output)
            logits3 = classifier3(pooled_output)

            l2_para = list(classifier2.parameters())
            l3_para = list(classifier3.parameters())
            
            l2_para_book = [x.detach().cpu().numpy() for x in l2_para]
            l3_para_book = [x.detach().cpu().numpy() for x in l3_para]
            l2_para_book = np.append(l2_para_book[0], l2_para_book[1])
            l3_para_book = np.append(l3_para_book[0], l3_para_book[1])

            wm2 = torch.autograd.Variable(torch.Tensor(l2_para_book).cuda(), requires_grad=True)
            wm3 = torch.autograd.Variable(torch.Tensor(l3_para_book).cuda(), requires_grad=True)

            orth_loss = torch.norm(wm2 * wm3)
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, model.num_labels), label_ids.view(-1))
            loss2 = loss_fct(logits2.view(-1, model.num_labels), label_ids.view(-1))
            loss3 = loss_fct(logits3.view(-1, model.num_labels), label_ids.view(-1))

            loss = loss1 + loss2 + loss3 +gamma*orth_loss
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                if turn == 1:
                    optimizer.backward(loss)
                elif turn == 2:
                    optimizer2.backward(loss)
                elif  turn == 3:
                    optimizer3.backward(loss)
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # print("loss", loss.item())
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                    args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                if turn == 1:
                    model.classifier = classifier1
                    optimizer.step()
                    optimizer.zero_grad()
                    classifier1 = model.classifier
                elif turn == 2:
                    model.classifier = classifier2
                    optimizer2.step()
                    optimizer2.zero_grad()
                    classifier2 = model.classifier
                    # print("After training:",list(classifier2.parameters()))
                elif turn == 3:
                    model.classifier = classifier3
                    optimizer3.step()
                    optimizer3.zero_grad()
                    classifier3 = model.classifier
                global_step += 1
        print("loss", tr_loss)
    model.classifier = classifier1
    return model, classifier2, classifier3


def TriTraining(model, classifier2, classifier3, args, device, n_gpu, epochs, processor, label_list, tokenizer):
    '''
        Tri-Trainint Process
    '''
    train_size = 300
     # initial the train set L
    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

    input_ids_train = np.array([f.input_ids for f in train_features])
    input_mask_train = np.array([f.input_mask for f in train_features])
    segment_ids_train = np.array([f.segment_ids for f in train_features])
    label_ids_train = np.array([f.label_id for f in train_features])
    train_data_loader = load_train_data(args, input_ids_train, input_mask_train, segment_ids_train, label_ids_train)

    # initial the unlabeled set U
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer
    )
    unlabeled_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    unlabeled_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    unlabeled_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    unlabeled_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(unlabeled_input_ids, unlabeled_input_mask, unlabeled_segment_ids, unlabeled_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Tri Training *****")
    cnt = 0
    for cnt in trange(2, 3 * epochs + 1, desc="Turn"):
        trainset_index = []
        model.eval()
        classifier2.eval()
        classifier3.eval()
        predict_results_j = []
        predict_results_k = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                pooled_output = model(input_ids, segment_ids, input_mask, mode=1)
                if cnt % 3 == 1:
                    logits_j = classifier2(pooled_output)
                    logits_k = classifier3(pooled_output)
                elif cnt % 3 == 2:
                    logits_j = model.classifier(pooled_output)
                    logits_k = classifier3(pooled_output)
                elif cnt % 3 == 0:
                    logits_j = model.classifier(pooled_output)
                    logits_k = classifier2(pooled_output)
            logits_j = logits_j.detach().cpu().numpy()
            logits_k = logits_k.detach().cpu().numpy()
            predict_results_j.extend(np.argmax(logits_j, axis=1))
            predict_results_k.extend(np.argmax(logits_k, axis=1))

        # choose  p2(x) == p3(x) 
        # print("predict_result_j", predict_results_j)
        # print("predict_result_k", predict_results_k)
        for i in range(len(predict_results_j)):
            if predict_results_j[i] == predict_results_k[i]:
                trainset_index.append(i)
            
            
        # doing the permutation
        permutation = np.array(trainset_index)
        print("permutation size ", len(permutation))
        if len(permutation) == 0:
            train_data_loader = load_train_data(args, input_ids_train, input_mask_train, segment_ids_train, label_ids_train)
        else:
            if cnt % 3 == 1:
                input_ids_train_new = unlabeled_input_ids[permutation]
                input_mask_train_new = unlabeled_input_mask[permutation]
                segment_ids_train_new = unlabeled_segment_ids[permutation]
                label_ids_train_new = np.array(predict_results_j)[permutation]
                train_data_loader = load_train_data(args, input_ids_train_new, input_mask_train_new, segment_ids_train_new, label_ids_train_new)
            else:
                # print("input_ids_train shape:", input_ids_train.shape)
                # print("unlabeled_input_ids shape", unlabeled_input_ids[permutation].shape)
                input_ids_train_new = np.concatenate((input_ids_train, unlabeled_input_ids[permutation]), axis=0)
                input_mask_train_new = np.concatenate((input_mask_train,unlabeled_input_mask[permutation]), axis=0)
                segment_ids_train_new = np.concatenate((segment_ids_train, unlabeled_segment_ids[permutation]), axis=0)
                label_ids_train_new = np.concatenate((label_ids_train, np.array(predict_results_j)[permutation]), axis=0)
                train_data_loader = load_train_data(args, input_ids_train_new, input_mask_train_new, segment_ids_train_new, label_ids_train_new)
        if cnt % 3 == 0:
            model, classifier2, classifier3 = train_model(model, classifier2, classifier3, args, n_gpu, train_data_loader, device, args.num_train_epochs, logger, 3)
        else:
            model, classifier2, classifier3 = train_model(model, classifier2, classifier3, args, n_gpu, train_data_loader, device, args.num_train_epochs, logger, cnt % 3)

    return model, classifier2, classifier3


def majority_vote(a, b, c):
    '''
        vote for the final res
    '''
    assert len(a) == len(b)
    assert len(a) == len(c)
    res = []

    for i in range(len(a)):
        if a[i] == b[i]:
            res.append(a[i])
        elif a[i] == c[i]:
            res.append(a[i])
        elif b[i] == c[i]:
            res.append(b[i])
        else:
            res.append(c[i])
    return np.array(res)
    

def evaluate_model(model, classifier2, classifier3, device, eval_data_loader, logger):
    model.eval()
    classifier2.eval()
    classifier3.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples =0, 0

    idx = 0
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader, desc="Iteration"):
        idx += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            pooled_output = model(input_ids, segment_ids, input_mask, mode=1)
            logits1 = model.classifier(pooled_output)
            logits2 = classifier2(pooled_output)
            logits3 = classifier3(pooled_output)
        
        logits1 = logits1.detach().cpu().numpy()
        logits2 = logits2.detach().cpu().numpy()
        logits3 = logits3.detach().cpu().numpy()
        outputs1 = np.argmax(logits1, axis=1)
        outputs2 = np.argmax(logits2, axis=1)
        outputs3 = np.argmax(logits3, axis=1)

        outputs = majority_vote(outputs1, outputs2, outputs3)
        label_ids = label_ids.to('cpu').numpy()
        # print(outputs.shape)
        # print(label_ids.shape)
        if idx < 4:
            logger.info('  prediction label = %s, reference label = %s', outputs, label_ids)
        
        tmp_eval_accuracy = np.sum(outputs == label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    
    eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("accuracy: %f", eval_accuracy)
    return eval_accuracy

    
    

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
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
    parser.add_argument("--selection_function",
                        type=str,
                        default= "random",
                        help = "choose the selectionfunction")
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
    parser.add_argument("--ft_both",
                        action="store_true",
                        help="fine-tune the student model with both true and pseudo data")
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

    logger.info("***** Build tri model *****")
    # Prepare model
    cache_dir = args.cache_dir
    
    model = create_tri_model(args, cache_dir, num_labels, device)



    if args.do_train:
        # step 0: load train examples
        logger.info("Cook training and dev data for teacher model")
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info(" Num Training Examples = %d", len(train_examples))
        logger.info(" Train Batch Size = %d", args.train_batch_size)

        input_ids_train = np.array([f.input_ids for f in train_features])
        input_mask_train = np.array([f.input_mask for f in train_features])
        segment_ids_train = np.array([f.segment_ids for f in train_features])
        label_ids_train = np.array([f.label_id for f in train_features])
        train_data_loader = load_train_data(args, input_ids_train, input_mask_train, segment_ids_train, label_ids_train)
        
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
        )
        logger.info(" Num Eval Examples = %d", len(eval_examples))
        logger.info(" Eval Batch Size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    

        
        # step 1: train the Tri model with labeled data
        logger.info("***** Running train TL model with labeled data *****")
        model = init_tri_model(model, args, n_gpu, train_data_loader, device, args.num_train_epochs, logger)

        # step 2: copy 2 classifiers from the Tri model, the default classifier is in the bert
        # classifier2 = copy.deepcopy(model.classifier)
        # classifier3 = copy.deepcopy(model.classifier)

        classifier2 = nn.Linear(model.classifier.in_features, num_labels).cuda()
        classifier3 = nn.Linear(model.classifier.in_features, num_labels).cuda()

        # step 3: Tri-training

        model, classifier2, classifier3 = TriTraining(model, classifier2, classifier3, args, device, n_gpu, 10, processor, label_list, tokenizer)
        
        # step 4: evalute model 
        acc = evaluate_model(model, classifier2, classifier3, device, eval_data_loader, logger)
        
if __name__ == "__main__":
    main()
