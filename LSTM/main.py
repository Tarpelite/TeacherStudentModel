import argparse
import os
import time
import logging 
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model import LSTMClassifier

from tqdm import tqdm, trange
from pytorch_pretrained_bert.teacher_student_model.processor_zoo import OOCLAUSProcessor, DBpediaProcessor, TrecProcessor, YelpProcessor
from pytorch_pretrained_bert.teacher_student_model.IO_utils import split, load_train_data, load_eval_data, convert_examples_to_features, sample_data
from pytorch_pretrained_bert.teacher_student_model.selection_zoo import RandomSelectionFunction, TopkSelectionFunction, BalanceTopkSelectionFunction


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO
)

logger = logging.getLogger(__name__)


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_dataloader, device, num_epoch, logger):
    '''
        train model
    '''
    logger.info("***** Initial optimizer *****")
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    
    logger.info("***** Running train *****")
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.to(device)
    model.train()
    for _ in trange(int(num_epoch), desc="Epoch"):
        tr_loss = 0
        nb_tr_example, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            optim.zero_grad()
            prediction = model(text)
            loss = loss_fn(prediction, label_ids)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0*num_corrects/len(batch)
            loss.backward()
            clip_gradient(model, 1e-1)
            optim.step()
            step += 1

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return model

def eval_model(model, eval_dataloader, device, logger):
    '''
        evaluater the model
    '''
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Iteration"):
        idx += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids)
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
    logger.info("accuracy: %f", eval_accuracy)
    return eval_accuracy

def create_model(batch_size, output_size, hidden_size, num_layers, embedding_length):

    model_new = LSTMClassifier(batch_size, output_size, hidden_size, num_layers, embedding_length)

    return model_new

def main():
    parser = argparse.ArgumentParser()

    #Required parameters
    parser.add_argument("--learning_rate",
                        type=float,
                        default=2e-5)
    parser.add_argument("--batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--hidden_size",
                        type=int,
                        default=256)
    parser.add_argument("--seq_len",
                        type=int,
                        default=512)
    parser.add_argument("--task_name",
                        type=str,
                        default="trec")
    parser.add_argument("--data_dir",
                        type=str)
    parser.add_argument("--num_train_epochs",
                        type=str,
                        default=10)
    args = parser.parse_args()
    
    logger.info("***** Building Models *****")
    model_TL = create_model(args.batch_size, args.output_size, args.hidden_size, args.num_layers, args.seq_len)
    model_TU = create_model(args.batch_size, args.output_size, args.hidden_size, args.num_layers, args.seq_len) 
    model_S = create_model(args.batch_size, args.output_size, args.hidden_size, args.num_layers, args.seq_len)

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

    processor = processors[args.task_name.lower()]
    num_labels = num_labels_task[args.task_name.lower()]
    label_list = processor.get_labels()

    logger.info("Cook training and dev data for teacher model")
    train_examples = processor.get_train_examples(args.data_dir)
    train_features=convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
    





        


