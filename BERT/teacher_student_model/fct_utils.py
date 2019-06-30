from tqdm import tqdm, trange
import torch
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
import numpy as np

def init_optimizer(model, args, data_loader):
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = data_loader.dataset
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
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

def KNN_train(model, train_dataloader, logger):
    '''
        train KNN model
    '''
    logger.info("***** Running train *****")
    for _ , batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple( t for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        X = np.concatenate((input_ids, input_mask, segment_ids), axis=1)
        Y = np.array(label_ids)
        model.fit(X, Y)

def KNN_predict(model, eval_dataloader, logger):
    '''
        KNN model predict
    '''
    predict_result = []

    for _,  batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        X = np.concatenate((input_ids, input_mask, segment_ids), axis=1)
        outputs = model.predict_proba(X)
        predict_result.extend(outputs)

    return predict_result
        

def KNN_evaluate(model, eval_dataloader, logger):
    '''
        evalute KNN model
    '''
    logger.info("***** Running evaluate *****")
    eval_acc = 0
    nb_eval_examples = 0
    for _,  batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        X = np.concatenate((input_ids, input_mask, segment_ids), axis=1)
        outputs = model.predict_proba(X)
        Y = np.array(label_ids)
        print(outputs)
        predict_labels = np.argmax(outputs, axis= 1)
        tmp_acc = np.sum(predict_labels == Y)
        eval_acc += tmp_acc
        nb_eval_examples += input_ids.size(0)

    eval_acc = eval_acc / nb_eval_examples

    print("accuracy:", eval_acc)
    return eval_acc




def train(model, args, n_gpu, train_dataloader, device, num_epoch, logger):
    '''
        train model
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
            loss = model(input_ids, segment_ids, input_mask, label_ids)
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
