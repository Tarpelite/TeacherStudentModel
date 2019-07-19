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
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling_for_doc import BertForDocMultiClassification
from oocl_utils.evaluate import evaluation_report
from teacher_student_model.processor_zoo import OOCLAUSProcessor, DBpediaProcessor, TrecProcessor, YelpProcessor
from teacher_student_model.fct_utils import train, init_optimizer, create_model, predict_model, evaluate_model, init_student_weights
from oocl_utils.score_output_2_labels import convert
from sklearn.neighbors import KNeighborsClassifier
from teacher_student_model.selection_zoo import RandomSelectionFunction, TopkSelectionFunction, BalanceTopkSelectionFunction
from teacher_student_model.IO_utils import split, load_train_data, load_eval_data, convert_examples_to_features, sample_data


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class TwinModel(nn.Module):
    def __init__(self,num_labels):
        super(TwinModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1, num_labels), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(1, num_labels), requires_grad=True)
        self.num_labels = num_labels
    
    def forward(self, cls1, cls2, labels=None):
        # cls1 = torch.Tensor(cls1).long()
        # cls2 = torch.Tensor(cls2).long()
        print("OK!")
        logits = self.w1 * cls1 + self.w2 * cls2
        print("cls1 shape", cls1.shape)
        print("cls2 shape", cls2.shape)
        print("logits shape",logits.shape)
        print("w1 shape", self.w1.shape)
        print("w2 shape", self.w2.shape)
        if labels is None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            print("loss shape", loss.shape)
            return loss
        else:
            return logits

def create_student_model(args, cache_dir, num_labels, device):
    '''
        create TwinModel as Student
    '''
    model_new = TwinModel(num_labels)
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

def init_student_optimizer(model, args, data_loader):
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = data_loader.dataset
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    print(model.state_dict().keys())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    weight_decay = ['w1','w2']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if  any(nd in n for nd in weight_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in weight_decay)], 'weight_decay': 0.0},
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

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
    return optimizer, num_train_optimization_steps

def train_student_model(model, args, n_gpu, train_dataloader, device, num_epoch, logger, model_TL, model_TU, w1=None, w2=None):
    '''
        train student model
    '''

    logger.info("***** Running train *****")
    model.train()
    model_TL.eval()
    model_TU.eval()
    learning_rate = 2e-1
    if w1 is None:
        w1 = nn.Parameter(torch.Tensor([0.5]*model.num_labels).cuda(), requires_grad=True)
    if w2 is None:
        w2 = nn.Parameter(torch.Tensor([0.5]*model.num_labels).cuda(), requires_grad=True)
    for _ in trange(int(num_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            cls1 = model_TL(input_ids, segment_ids, input_mask)
            cls2 = model_TU(input_ids, segment_ids, input_mask)
            y = w1 * cls1 + w2 * cls2
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(y.view(-1, model.num_labels), label_ids.view(-1))
            tr_loss += loss.item()
            
            '''
            print("cls1:", cls1)
            print("cls1_type:", type(cls1))
            print("cls1_size:", cls1.size)
            print("cls2:", cls2)
            print("cls2_size:", cls2.size)
            print("cls2_type:", type(cls2))
            print("labels", label_ids)
            print("labels_type", type(label_ids))
            print("labels_size", label_ids.size)
            '''
            # loss = model(cls1, cls2, label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            w1.data -= learning_rate * w1.grad.data
            w2.data -= learning_rate * w2.grad.data
            w1.grad.data.zero_()
            w2.grad.data.zero_()
        print("loss", loss)
        print("w1:{} w2:{}".format(w1, w2))

    return model, w1, w2

def train_student_model_with_attention(model, args, n_gpu, train_dataloader, device, num_epoch, logger, model_TL, model_TU, w1=None, w2=None):
    logger.info("***** Running train *****")
    model.train()
    model_TL.eval()
    model_TU.eval()
    learning_rate = 2e-2
    alpha1 = None
    alpha2 = None
    if w1 is None:
        w1 = nn.Parameter(torch.Tensor([0.5]*model.num_labels).cuda(), requires_grad=True)
    if w2 is None:
        w2 = nn.Parameter(torch.Tensor([0.5] * model.num_labels).cuda(), requires_grad=True)
    
    for _ in trange(int(num_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            cls1 = model_TL(input_ids, segment_ids, input_mask)
            cls2 = model_TU(input_ids, segment_ids, input_mask)
            alpha1 = w1 * cls1
            alpha2 = w2 * cls2
            y = torch.exp(alpha1)*cls1/(torch.exp(alpha1) + torch.exp(alpha2)) + torch.exp(alpha2)*cls2/(torch.exp(alpha1) + torch.exp(alpha2))
            # y = w1 * cls1 + w2 * cls2
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(y.view(-1, model.num_labels), label_ids.view(-1))
            tr_loss += loss.item()
            
            '''
            print("cls1:", cls1)
            print("cls1_type:", type(cls1))
            print("cls1_size:", cls1.size)
            print("cls2:", cls2)
            print("cls2_size:", cls2.size)
            print("cls2_type:", type(cls2))
            print("labels", label_ids)
            print("labels_type", type(label_ids))
            print("labels_size", label_ids.size)
            '''
            # loss = model(cls1, cls2, label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            w1.data -= learning_rate * w1.grad.data
            w2.data -= learning_rate * w2.grad.data
            w1.grad.data.zero_()
            w2.grad.data.zero_()
        print("loss", loss)
        print("w1:{} w2:{}".format(w1, w2))

    return model, w1, w2

def evaluate_student_model(model_TL, model_TU, w1, w2, device, eval_data_loader, logger):
    model_TL.eval()
    model_TU.eval()
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
            cls1 = model_TL(input_ids, segment_ids, input_mask)
            cls2 = model_TU(input_ids, segment_ids, input_mask)
            alpha1 = w1 * cls1
            alpha2 = w2 * cls2
            logits = torch.exp(alpha1)*cls1/(torch.exp(alpha1) + torch.exp(alpha2)) + torch.exp(alpha2)*cls2/(torch.exp(alpha1) + torch.exp(alpha2))
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

def evaluate_student_model_with_attention(model_TL, model_TU, w1, w2, device, eval_data_loader, logger):
    model_TL.eval()
    model_TU.eval()
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
            cls1 = model_TL(input_ids, segment_ids, input_mask)
            cls2 = model_TU(input_ids, segment_ids, input_mask)
            logits = w1 * cls1 + w2 * cls2
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
                        help="the weights of the TL model in the final model")
    parser.add_argument("--with_attention",
                        action="store_true",
                        help="define the student train function")
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

    logger.info("***** Build teacher(label) model *****")
    # Prepare model
    cache_dir = args.cache_dir
    model_TL  = create_model(args, cache_dir, num_labels, device)

    logger.info("***** Build teacher(unlabel) model *****")
    cache_dir = args.cache_dir
    model_TU  = create_model(args, cache_dir, num_labels, device)
    logger.info("***** Build student model *****")

    model_student = create_student_model(args, cache_dir, num_labels, device)
    
    logger.info("***** Finish TL, TU model building *****")


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
            selection = BalanceTopkSelectionFunction()
        else:
            selection = TopkSelectionFunction()
        permutation = selection.select(probas_val, args.top_k)
        input_ids_TU, input_mask_TU, segment_ids_TU, label_ids_TU = sample_data(all_input_ids, all_input_mask, all_segment_ids, probas_val, permutation)
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
        # model_student = create_student_model(args, cache_dir, num_labels, device, model_TL, model_TU)
        model_student.to(device)

        # mix train data and pesudo data to create fine-tune dataset
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        input_ids_train = np.array([f.input_ids for f in train_features])
        input_mask_train = np.array([f.input_mask for f in train_features])
        segment_ids_train = np.array([f.segment_ids for f in train_features])
        label_ids_train = np.array([f.label_id for f in train_features])

        input_ids_ft = np.concatenate((input_ids_train, np.array(input_ids_TU)), axis=0)
        input_mask_ft = np.concatenate((input_mask_train, np.array(input_mask_TU)), axis=0)
        segment_ids_ft = np.concatenate((segment_ids_train, np.array(segment_ids_TU)), axis=0)
        label_ids_ft = np.concatenate((label_ids_train, np.array(label_ids_TU)), axis=0)

        p = np.random.permutation(len(input_ids_ft))
        input_ids_ft = input_ids_ft[p]
        input_mask_ft = input_mask_ft[p]
        segment_ids_ft = segment_ids_ft[p]
        label_ids_ft = label_ids_ft[p]

        fine_tune_dataloader = load_train_data(args, input_ids_ft, input_mask_ft, segment_ids_ft, label_ids_ft)
        if args.with_attention:
            train_student = train_student_model_with_attention
        else:
            train_student = train_student_model
        if args.ft_true:
            # step 6: train student model with train data
            logger.info("***** Running train student model with train data *****")
            model_student, w1, w2 = train_student(model_student, args, n_gpu, train_data_loader, device, args.num_student_train_epochs, logger, model_TL, model_TU)
            model_student.to(device)

        if args.ft_pseudo:
            # step 7: train student model with Pseudo labels
            logger.info("***** Running train student model with Pseudo data *****")
            model_student, w1, w2 = train_student(model_student, args, n_gpu, train_data_loader_TU, device, args.num_student_train_epochs, logger, model_TL, model_TU)
            model_student.to(device)
        
        if args.ft_both:
            # step 8: train student model with both train and Peudo data
            logger.info("***** Running train student model with both train and Pseudo data *****")
            model_student,w1, w2 = train_student(model_student, args, n_gpu, fine_tune_dataloader, device, args.num_student_train_epochs, logger, model_TL, model_TU)
            model_student.to(device)
            
        
        logger.info("***** Evaluate student model  *****")
        model_student_accuracy = evaluate_student_model(model_TL, model_TU, w1, w2, device, eval_data_loader, logger)
    

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
