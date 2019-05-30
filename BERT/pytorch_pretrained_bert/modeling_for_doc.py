"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MultiLabelSoftMarginLoss, BCELoss
import torch

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm, BertPreTrainedModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForDocClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    This module is to handle the long sequence (Length is more than 512).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels, max_seq_length):
        super(BertForDocClassification, self).__init__()
        bert_max_length = config.max_position_embeddings
        if bert_max_length > max_seq_length:
            logger.warning("Bert max length > max seq length, use BertForSequenceClassification instead of this")
        self.bert_max_length = bert_max_length
        self.bert = BertModel(config)
        self.document_att = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(-1, self.bert_max_length)
        token_type_ids = token_type_ids.view(-1, self.bert_max_length)
        attention_mask = attention_mask.view(-1, self.bert_max_length)

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = pooled_output.view(batch_size, -1)
        pooled_output_size = pooled_output.size(1)
        part1 = self.document_att(pooled_output[:, :pooled_output_size // 2])
        part2 = self.document_att(pooled_output[:, pooled_output_size // 2:])
        # pooled_output = pooled_output[:, : pooled_output_size // 2] + pooled_output[:, pooled_output_size // 2:]
        document_att_weight = nn.Softmax()(torch.cat((part1, part2), 1)).unsqueeze(2)
        pooled_output = torch.matmul(pooled_output.view(-1, pooled_output_size //2, 2), document_att_weight).squeeze()
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForDocMultiClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    This module is to handle the long sequence (Length is more than 512).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForDocMultiClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        if labels is not None:
            # loss_fct = MultiLabelSoftMarginLoss()
            loss_fct = BCELoss()
            labels = labels.float()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return logits


class BertForDocMTPSMultiClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    This module is to handle the long sequence (Length is more than 512).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForDocMTPSMultiClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        if labels is not None:
            # loss_fct = MultiLabelSoftMarginLoss()
            loss_fct = BCELoss()
            labels = labels.float()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return logits