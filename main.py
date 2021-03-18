from bert import *
from config import Config
import logging
from evaluation import CyclicLR
from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from tqdm import tqdm_notebook as tqdm
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
args = Config().args


def deleteDuplicatedElementFromList(listA):
  return sorted(set(listA), key = listA.index)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    xxxx = 1
    for (ex_index, example) in enumerate(examples):
        tokens_a = deleteDuplicatedElementFromList(tokenizer.tokenize(example.text))
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

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

        labels_ids = []
        for label in example.labels:
            labels_ids.append(float(label))

        #      label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=labels_ids))
    return features


def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs == labels)


def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
  "Compute accuracy when `y_pred` and `y_true` are the same size."
  if sigmoid: y_pred = y_pred.sigmoid()
  if args["no_cuda"]:
    return np.mean(((y_pred>thresh)==y_true.byte()).float().numpy(), axis=1).sum()
  return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
  "Computes the f_beta between `preds` and `targets`"
  beta2 = beta ** 2
  if sigmoid: y_pred = y_pred.sigmoid()
  y_pred = (y_pred>thresh).float()
  y_true = y_true.float()
  TP = (y_pred*y_true).sum(dim=1)
  prec = TP/(y_pred.sum(dim=1)+eps)
  rec = TP/(y_true.sum(dim=1)+eps)
  res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
  return res.mean().item()


def warmup_linear(x, warmup=0.002):
  if x < warmup:
    return x/warmup
  return 1.0 - x


processors = {
  "toxic_multilabel": MultiLabelTextProcessor
}

# Setup GPU parameters

if args["local_rank"] == -1 or args["no_cuda"]:
  # device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(args['local_rank'])
  device = torch.device("cuda", args['local_rank'])
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))

args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

task_name = args['task_name'].lower()

if task_name not in processors:
  raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name](args['data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)

model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels=103)
model.cuda()

tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])

train_examples = None
num_train_steps = None
if args['do_train']:
  train_examples = processor.get_train_examples()
#  train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
  num_train_steps = int(
    len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])

  model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels)

  if args['fp16']:
      model.half()
  model.to(device)
  if args['local_rank'] != -1:
      try:
          from apex.parallel import DistributedDataParallel as DDP
      except ImportError:
          raise ImportError(
              "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
      model = DDP(model)
  elif n_gpu > 1:
      model = torch.nn.DataParallel(model)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
t_total = num_train_steps
if args['local_rank'] != -1:
  t_total = t_total // torch.distributed.get_world_size()
if args['fp16']:
  try:
    from apex.optimizers import FP16_Optimizer
    from apex.optimizers import FusedAdam
  except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

  optimizer = FusedAdam(optimizer_grouped_parameters,
                        lr=args['learning_rate'],
                        bias_correction=False,
                        max_grad_norm=1.0)
  if args['loss_scale'] == 0:
    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
  else:
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args['loss_scale'])
else:
  optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         t_total=t_total)

scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)

# Eval Fn
eval_examples = processor.get_dev_examples()


def eval():
    args['output_dir'].mkdir(exist_ok=True)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    all_logits = None
    all_labels = None

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            if args["no_cuda"]:
                all_logits = logits.detach().numpy()
            else:
                all_logits = logits.detach().cpu().numpy()
        else:
            if args["no_cuda"]:
                all_logits = np.concatenate((all_logits, logits.detach().numpy()), axis=0)
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            if args["no_cuda"]:
                all_labels = label_ids.detach().numpy()
            else:
                all_labels = label_ids.detach().cpu().numpy()
        else:
            if args["no_cuda"]:
                all_labels = np.concatenate((all_labels, label_ids.detach().numpy()), axis=0)
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    pred = np.where(all_logits > 0, 1.0, 0.0)
    # f1_score_macro = f1_score(all_labels, pred, average='macro', zero_division=1)
    # f1_score_micro = f1_score(all_labels, pred, average='micro', zero_division=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'roc_auc': roc_auc,
              'micro': roc_auc["micro"]}
    # 'f1_macro': f1_score_macro,
    # 'f1_micro': f1_score_micro}

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result


def fit(num_epocs=args['num_train_epochs']):
  global_step = 0
  model.train()
  for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
      # batch = tuple(t.type(torch.cuda.LongTensor) for t in batch)
      batch = tuple(t.to(device) for t in batch)
      input_ids, input_mask, segment_ids, label_ids = batch
      loss = model(input_ids, segment_ids, input_mask, label_ids)
      if n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
      if args['gradient_accumulation_steps'] > 1:
        loss = loss / args['gradient_accumulation_steps']

      if args['fp16']:
        optimizer.backward(loss)
      else:
        loss.backward()

      tr_loss += loss.item()
      nb_tr_examples += input_ids.size(0)
      nb_tr_steps += 1
      if (step + 1) % args['gradient_accumulation_steps'] == 0:
        lr_this_step = args['learning_rate'] * warmup_linear(global_step/t_total, args['warmup_proportion'])
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
    logger.info('Eval after epoc {}'.format(i_+1))
    eval()

train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args['train_batch_size'])
logger.info("  Num steps = %d", num_train_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args['local_rank'] == -1:
  train_sampler = RandomSampler(train_data)
else:
  train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

if __name__ == "__main__":
    fit()