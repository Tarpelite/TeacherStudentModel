import numpy as np
import collections
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseSelectionFunction(object):
    def __init__(self):
        return
    
    def __select(self):
        return

class RandomSelectionFunction(object):
    '''
        randomly choose nums from the sample
    '''
    @staticmethod
    def select(probas_val, samples):
        selection = np.random.choice(probas_val.shape[0], samples, replace=False)

        return selection

class TopkSelectionFunction(object):
    '''
        choose the topk data from the predict data
    '''
    @staticmethod
    def select(probas_val, topk):
        label_ids_predict = np.array(probas_val)
        bitmap = np.zeros(len(probas_val))
        class_num = label_ids_predict.shape[1]
        permutation = []
        for i in range(class_num):
            pos_sort = sorted(range(len(probas_val)), key=lambda k: probas_val[k][i], reverse=True)
            pos_sort = pos_sort[:topk]
            for pos in pos_sort:
                bitmap[pos] = 1
        for i in range(len(bitmap)):
            if bitmap[i] == 1:
                permutation.append(i)
        permutation = np.array(permutation)
        return permutation

class BalanceTopkSelectionFunction(object):
    '''
        the distribution of classes is balanced and reach the num of top k.
        try best to balance.
    '''
    @staticmethod
    def select(probas_val, topk):
        logger.info(" topk = %d", topk)
        probas_val = np.array(probas_val)
        logger.info(" probas_val shape %d %d", probas_val.shape[0], probas_val.shape[1])
        labels = np.argmax(probas_val, axis=1)
        logger.info("label distribution = %s", collections.Counter(labels))
        classes = len(probas_val[0])
        pos = 0
        classes_unsort = [[] for x in range(classes)]
        permutation = []
        
        for label in labels:
            row = [pos, probas_val[pos][label]]
            classes_unsort[label].append(row)
            pos += 1
        
        for i in range(classes):
            class_i = classes_unsort[i]
            class_i.sort(key=lambda k: k[1], reverse=True)
            class_len = len(class_i)
            if class_len < topk:
                for row in class_i:
                    permutation.append(row[0])
            else:
                for row in class_i:
                    permutation.append(row[0])
        permutation = np.array(permutation)
        return permutation
