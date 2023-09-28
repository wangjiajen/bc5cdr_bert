import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel,BertTokenizer,BertForTokenClassification,BertTokenizerFast
from datasets import load_dataset
from transformers import BertForTokenClassification, BertTokenizer, AdamW,AutoTokenizer,AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bert
tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)

# 加載數據集
dataset = load_dataset("tner/bc5cdr")
train_data = dataset["train"]
val_data = dataset["validation"]
# df = pd.DataFrame(train_dataset)
# print(df)
# print(dataset)

# Dataset
class Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["tokens"]
        label = self.data[idx]["tags"]
        
        # print(text)
        # print(label)
        return text, label

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    # print(f"labels: {labels}, word_ids: {word_ids}")
    # print(word_ids)
    # print(len(word_ids))
    # print(labels)
    # print(len(labels))
    for word_id in word_ids:
        
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = -100
                new_labels.append(label)      
        # print(new_labels)


    return new_labels








# def split_entity(label_sequence):
#     entity_mark = dict()
#     entity_pointer = None
#     for index, label in enumerate(label_sequence):
#         if label.startswith('B'):
#             category = label.split('-')[1]
#             entity_pointer = (index, category)
#             entity_mark.setdefault(entity_pointer, [label])
#         elif label.startswith('I'):
#             if entity_pointer is None:
#                 continue
#             if entity_pointer[1] != label.split('-')[1]:
#                 continue
#             entity_mark[entity_pointer].append(label)
#         else:
#             entity_pointer = None
#     return entity_mark


# def evaluate(real_label, predict_label):
    
#     real_entity_mark = split_entity(real_label)
#     predict_entity_mark = split_entity(predict_label)

#     true_entity_mark = dict()
#     key_set = real_entity_mark.keys() & predict_entity_mark.keys()
#     for key in key_set:
#         real_entity = real_entity_mark.get(key)
#         predict_entity = predict_entity_mark.get(key)
#         if tuple(real_entity) == tuple(predict_entity):
#             true_entity_mark.setdefault(key, real_entity)

#     real_entity_num = len(real_entity_mark)
#     predict_entity_num = len(predict_entity_mark)
#     true_entity_num = len(true_entity_mark)

#     precision = true_entity_num / predict_entity_num
#     recall = true_entity_num / real_entity_num
#     f1 = 2 * precision * recall / (precision + recall)

#     return precision, recall, f1

def collate_fn(batch):
    texts, labels = zip(*batch)

    # 標記文本
    tokenized_inputs = tokenizer(
        texts, truncation=True, is_split_into_words=True, padding=True, max_length=128
    ,return_tensors="pt")
    # print(texts)
    # print(labels)
    # print(tokenized_inputs.input_ids)
    # print(tokenized_inputs.attention_mask)
    # 處理標籤
    new_labels = []
    for i, label_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(label_list, word_ids))
    # print(new_labels)

    # 將輸入和標籤轉換為PyTorch張量
    # input_ids = torch.tensor(tokenized_inputs.input_ids)
    # attention_mask = torch.tensor(tokenized_inputs.attention_mask)
    labels_input = torch.tensor(new_labels)
    # print(labels_input)
    return tokenized_inputs, labels_input

# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(
#         examples["tokens"], truncation=True, is_split_into_words=True
#     )
#     all_labels = examples["tags"]
#     new_labels = []
#     for i, labels in enumerate(all_labels):
#         word_ids = tokenized_inputs.word_ids(i)
#         new_labels.append(align_labels_with_tokens(labels, word_ids))

#     tokenized_inputs["labels"] = new_labels
#     return tokenized_inputs

# 
# tokenized_datasets = dataset.map(
#     tokenize_and_align_labels,
#     batched=True,
# )

# 打印
# print("Number of examples:", len(tokenized_datasets))
# tokenized_example = tokenized_datasets["train"][0]
# example = dataset["train"][0]
# tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
# print(tokens)
# 打印 tokenized_example
# print(tokenized_example)

# 定義參數
LR = 1e-6
EPOCHS = 5 #總共要用全部的訓練樣本重複跑幾回合
BATCH_SIZE = 8

train_dataset = Dataset(train_data,tokenizer)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)
val_dataset = Dataset(val_data,tokenizer)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)

# Define the model, optimizer, and loss function
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
model.train()
# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        # data=[t.to(device) for t in batch]
        # a,labels_input=data[:]
        # input_ids=a.input_ids
        # attention_mask=a.attention_mask
        # optimizer.zero_grad()
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels_input)
        
        inputs, labels_input = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels_input.to(device)
        optimizer.zero_grad()

        outputs = model(**inputs)
        logits = outputs.logits

        # Reshape logits and labels to [batch_size * max_seq_length, num_labels]
        logits = logits.view(-1, model.config.num_labels)
        labels =labels.view(-1)

        # Compute the loss
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {average_loss:.4f}')

    # Validation loop
    model.eval()
    val_true_labels = []
    val_pred_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
            inputs, labels_input = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels_input.to(device)

            # data=[t.to(device) for t in batch]
            # a,labels_input=data[:]
            # input_ids=a.input_ids
            # attention_mask=a.attention_mask
            
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels_input)
            outputs = model(**inputs)
            logits = outputs.logits

            # Reshape logits and labels to [batch_size * max_seq_length, num_labels]
            logits = logits.view(-1, model.config.num_labels)
            labels = labels.view(-1)

            _, predicted_labels = torch.max(logits, 1)

            val_true_labels.extend(labels.cpu().numpy())
            val_pred_labels.extend(predicted_labels.cpu().numpy())

    # Calculate evaluation metrics
    precision = precision_score(val_true_labels, val_pred_labels, average='macro')
    recall = recall_score(val_true_labels, val_pred_labels, average='macro')
    f1 = f1_score(val_true_labels, val_pred_labels, average='macro')

    print(f'Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')



