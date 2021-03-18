from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from fastai.text import Tokenizer, Vocab
import pandas as pd
import collections
import os
import pdb
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
  def __init__(self, text, labels=None):
    self.text = text
    self.labels = labels


class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


class DataProcessor(object):
    def get_train_examples(self):
        raise NotImplementedError()

    def get_dev_examples(self):
        raise NotImplementedError()

    def get_test_examples(self):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


class MultiLabelTextProcessor(DataProcessor):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def parse_label(self, label):
        datafile = open('RCV1/target_names.txt', 'r')
        target_names = np.array([line.strip().split(' ') for line in datafile][0], dtype=object)
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=target_names)
        data_label = mlb.fit_transform(label)
        return data_label.tolist()

    def get_train_examples(self):
        filename = 'train_text.txt'
        logger.info("LOOKING AT {}".format(os.path.join(self.data_dir, filename)))
        # if size == -1:
        data_df_file = open(os.path.join(self.data_dir, filename))
        data_df = [line.strip() for line in data_df_file]
        data_df = data_df[0:int(0.8 * len(data_df))]
        data_label = open(os.path.join(self.data_dir, 'train_label.dat'))
        label = [line.strip().split(' ') for line in data_label]
        label = label[0:int(0.8 * len(label))]
        label = self.parse_label(label)
        return self._create_examples(data_df, label, "train")

    def get_dev_examples(self):
        filename = 'train_text.txt'
        # if size == -1:
        data_df_file = open(os.path.join(self.data_dir, filename))
        data_df = [line.strip() for line in data_df_file]
        data_df = data_df[int(0.8 * len(data_df)):len(data_df)]
        data_label = open(os.path.join(self.data_dir, 'train_label.dat'))
        label = [line.strip().split(' ') for line in data_label]
        label = label[int(0.8 * len(label)):len(label)]
        label = self.parse_label(label)
        return self._create_examples(data_df, label, "dev")

    def get_test_examples(self):
        filename = 'test_text.txt'
        # if size == -1:
        data_df_file = open(os.path.join(self.data_dir, filename))
        data_df = [line.strip() for line in data_df_file]
        data_label = open(os.path.join(self.data_dir, 'test_label.dat'))
        label = [line.strip().split(' ') for line in data_label]
        label = self.parse_label(label)
        return self._create_examples(data_df, label, "test")

    def get_labels(self):
        if self.labels == None:
            label_file = open('RCV1/target_names.txt', 'r')
            self.labels = [line.strip().split(' ') for line in label_file][0]
        return self.labels

    def _create_examples(self, data, label, set_type, labels_available=True):
        examples = []
        for i in range(len(data)):
            text = data[i]
            if labels_available:
                labels = label[i]
            else:
                labels = []
            examples.append(InputExample(text=text, labels=labels))
        return examples


from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a CNN layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=103):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.num_channels = config.hidden_size // 3
        self.bert = BertModel(config)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(2, config.hidden_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(3, config.hidden_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(4, config.hidden_size))

        self.pool1 = nn.MaxPool1d(kernel_size=config.max_position_embeddings - 2 + 1)
        self.pool2 = nn.MaxPool1d(kernel_size=config.max_position_embeddings - 3 + 1)
        self.pool3 = nn.MaxPool1d(kernel_size=config.max_position_embeddings - 4 + 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = pooled_output.unsqueeze(1)

        h1 = self.conv1(pooled_output)
        h2 = self.conv2(pooled_output)
        h3 = self.conv3(pooled_output)

        h1 = self.pool1(h1.squeeze())
        h2 = self.pool2(h2.squeeze())
        h3 = self.pool3(h3.squeeze())

        pooled_output = torch.cat([h1, h2, h3], 1).squeeze()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

