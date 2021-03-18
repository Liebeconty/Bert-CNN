import os

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Predictions import generate_predictions
from PreprocessDate import plot_sentence_embeddings_length, tokenize_inputs, create_attn_masks
from trainXLNet import save_model, load_model, trainXLNet
from XLNet import *

print("GPU Available: {}".format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
print("Number of GPU Available: {}".format(n_gpu))
print("GPU: {}".format(torch.cuda.get_device_name(0)))

train = pd.read_csv('RCV1/train_data.csv', index_col='id')
test = pd.read_csv('RCV1/val_data.csv', index_col='id')[['comment_text']]


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
train_text_list = train["comment_text"].values
test_text_list = test["comment_text"].values

plot_sentence_embeddings_length(train_text_list, tokenizer)
plot_sentence_embeddings_length(test_text_list, tokenizer)


# create input id tokens / train
train_input_ids = tokenize_inputs(train_text_list, tokenizer, num_embeddings=250)
# create input id tokens / test
test_input_ids = tokenize_inputs(test_text_list, tokenizer, num_embeddings=250)
# create attention masks / train
train_attention_masks = create_attn_masks(train_input_ids)
# create attention masks / test
test_attention_masks = create_attn_masks(test_input_ids)
# add input ids and attention masks to the dataframe
train["features"] = train_input_ids.tolist()
train["masks"] = train_attention_masks

test["features"] = test_input_ids.tolist()
test["masks"] = test_attention_masks

# train valid split
train, valid = train_test_split(train, test_size=0.2, random_state=42)

X_train = train["features"].values.tolist()
X_valid = valid["features"].values.tolist()

train_masks = train["masks"].values.tolist()
valid_masks = valid["masks"].values.tolist()

label_cols = ["C11", "C12", "C13", "C14", "C15",
       "C151", "C1511", "C152",
       "C16", "C17", "C171",
       "C172", "C173", "C174",
       "C18", "C181", "C182",
       "C183", "C21", "C22",
       "C23", "C24", "C31",
       "C311", "C312", "C313",
       "C32", "C33", "C331",
       "C34", "C41", "C411",
       "C42", "CCAT", "E11",
       "E12", "E121", "E13",
       "E131", "E132", "E14",
       "E141", "E142", "E143",
       "E21", "E211", "E212",
       "E31", "E311", "E312",
       "E313", "E41", "E411",
       "E51", "E511", "E512",
       "E513", "E61", "E71",
       "ECAT",  "G15", "G151",
       "G152", "G153", "G154",
       "G155", "G156", "G157",
       "G158", "G159", "GCAT",
       "GCRIM", "GDEF", "GDIP",
       "GDIS", "GENT", "GENV",
       "GFAS", "GHEA", "GJOB",
       "GMIL", "GOBIT", "GODD", "GPOL", "GPRO", "GREL", "GSCI", "GSPO", "GTOUR", "GVIO", "GVOTE", "GWEA", "GWELF", "M11", "M12", "M13", "M131", "M132", "M14", "M141", "M142", "M143", "MCAT"]
Y_train = train[label_cols].values.tolist()
Y_valid = valid[label_cols].values.tolist()

# Convert all of our input ids and attention masks into
# torch tensors, the required datatype for our model

X_train = torch.tensor(X_train)
X_valid = torch.tensor(X_valid)

Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_valid = torch.tensor(Y_valid, dtype=torch.float32)

train_masks = torch.tensor(train_masks, dtype=torch.long)
valid_masks = torch.tensor(valid_masks, dtype=torch.long)

# Select a batch size for training
batch_size = 8

# Create an iterator of our data with torch DataLoader. This helps save on
# memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(X_train, train_masks, Y_train) # 封装成tensor的数据集，每一个样本都通过索引张量来获得
train_sampler = RandomSampler(train_data) # 无放回地随机采样样本元素
train_dataloader = DataLoader(train_data,\
                              sampler=train_sampler,\
                              batch_size=batch_size) # 数据加载器。组合了一个数据集和采样器，并提供关于数据的迭代器。

validation_data = TensorDataset(X_valid, valid_masks, Y_valid)
validation_sampler = SequentialSampler(validation_data) # 顺序采样样本，始终按照同一个顺序
validation_dataloader = DataLoader(validation_data,\
                                   sampler=validation_sampler,\
                                   batch_size=batch_size)

model = XLNetForMultiLabelSequenceClassification(num_labels=len(Y_train[0]))
# model = torch.nn.DataParallel(model)
# model.cuda()

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=False)
#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler

num_epochs = 3
cwd = os.getcwd()
model_save_path = output_model_file = os.path.join(cwd, "RCV1/.output/xlnet_toxic.bin")
model, train_loss_set, valid_loss_set = trainXLNet(model=model,
                                              num_epochs=num_epochs,
                                              optimizer=optimizer,
                                              train_dataloader=train_dataloader,
                                              valid_dataloader=validation_dataloader,
                                              model_save_path=model_save_path,
                                              device="cuda")
# Plot loss
num_epochs = np.arange(len(train_loss_set))

fig, ax = plt.subplots(figsize=(10, 5));
ax.plot(num_epochs, np.array(train_loss_set), label="Train Loss")
ax.plot(num_epochs, np.array(valid_loss_set), 'g-', label="Valid Loss")
#ax1.plot(episode_record, lose_record, 'r-', label="Lose %")
ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Number of Epochs")


num_labels = len(label_cols)
pred_probs = generate_predictions(model, test, num_labels, device="cuda", batch_size=32)
pred_probs

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

test["toxic"] = pred_probs[:,0]
test["severe_toxic"] = pred_probs[:,1]
test["obscene"] = pred_probs[:,2]
test["threat"] = pred_probs[:,3]
test["insult"] = pred_probs[:,4]
test["identity_hate"] = pred_probs[:,5]

test_to_csv = test.reset_index()
print(test_to_csv.head())