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
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
df = pd.DataFrame(train_dataset)
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
    print(labels_input)
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

train_dataset = Dataset(train_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)
val_dataset = Dataset(val_dataset, tokenizer)
val_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)

model.train()

# 定義損失函數
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)
model.to(device)
# # 訓練
def train(model, train_loader, val_loader, epochs, criterion, optimizer,save_path):
    best_val_acc = 0.0  
    for epoch in range(epochs):
        # model.train()
        total_loss_train = 0
        total_correct_train = 0

        for batch in tqdm(train_loader):
            # for t in batch["input_ids"]:
            #     print(t)
            # return 
            # print("EEEEEEEEEEEEEEEEE",batch)
            # input_ids = [t.to(device) for t in batch[0]]
            # attention_mask = [t.to(device) for t in batch[1]]
            # labels = [t.to(device) for t in a]           
            data=[t.to(device) for t in batch]
            a,labels_input=data[:]
            input_ids=a.input_ids
            attention_mask=a.attention_mask
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels_input)
            # logits = outputs.logits
            # loss = criterion(logits, labels_input)
            loss = criterion(outputs, labels_input)
            # loss=outputs[0]
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_correct_train += (outputs.argmax(1) == labels_input).sum().item()

        train_loss = total_loss_train / len(train_loader)
        train_acc = total_correct_train / len(train_loader.dataset)

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Best model saved!")

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels_input"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            
    all_preds.extend(outputs.argmax(1).cpu().numpy().tolist())
    all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / len(data_loader.dataset)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, avg_acc, precision, recall, f1

best_model_path = "best_model.pth"
train(model, train_loader, val_loader, EPOCHS, criterion, optimizer,best_model_path)


