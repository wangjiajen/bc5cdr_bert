import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seqeval.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 檢查是否有可用的GPU，如果有就使用GPU，否則使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Bert
tokenizer =  AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

# 加載數據集
dataset = load_dataset("tner/bc5cdr")
train_data = dataset["train"]
val_data = dataset["validation"]
df = pd.DataFrame(val_data)
# print(df)
# print(dataset)


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        texts = self.data[idx]["tokens"]
        labels = self.data[idx]["tags"]
        
        # print(text)
        # print(label)
        return texts, labels
    
mapping = {
    0: 'O',
    1: "B-Chemical",
    2: "B-Disease",
    3: "I-Disease",
    4: "I-Chemical"
}

def coffate_fn(examples):
    texts, all_labels = [], []
    for text, labels in examples:
        texts.append(text)
        all_labels.append(labels)

    # 標記文本
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, is_split_into_words=True,
                                 max_length=128, return_tensors="pt")
    # print(texts)
    # print(labels)
    # print(tokenized_inputs.input_ids)
    # print(tokenized_inputs.attention_mask)

    # 處理標籤
    targets = []
    for i, labels in enumerate(all_labels):
        label_ids = []
        for word_idx in tokenized_inputs.word_ids(batch_index=i):         
            if word_idx is None:
                label_ids.append(-100)
            else:                
                label_ids.append(labels[word_idx])
        targets.append(label_ids)

    # 將輸入和標籤轉換為PyTorch張量
    targets = torch.tensor(targets)
    return tokenized_inputs, targets

# 定義參數
#總共要用全部的訓練樣本重複跑幾回合
EPOCHS = 10 
BATCH_SIZE = 8

train_dataset = CustomDataset(train_data,tokenizer)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=coffate_fn)
val_dataset = CustomDataset(val_data,tokenizer)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=coffate_fn)

# Define the model, optimizer, and loss function
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
model.train()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss_train = 0.0


    # 訓練數據的batch
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):

        # 對batch中的每條tensor類型數據都執行.to(device)
        # 因為模型和數據要在同一個設備上才能運行
        inputs, targets = [x.to(device) for x in batch]

        # 確保模型輸出和目標形狀對齊
        labels = targets.view(-1)

        # 清除現有梯度
        optimizer.zero_grad()
        outputs = model(**inputs,labels=labels)
        logits = outputs.logits

        # 計算損失
        loss = criterion(logits.view(-1, 5), labels)
        loss.backward()
        optimizer.step()
        # logits_shape= logits.size()
        # print("模型输出 (logits) 的形状:", logits_shape)
        # print(f"inputs: {inputs}")
        # print(f"labels_input: {labels_input}")

        # total_loss_train += loss.item()

    # average_loss = total_loss_train / len(train_loader)
    # print(f'Epoch {epoch+1}, Loss: {average_loss:.4f}')
    model.save_pretrained('./saved_models')

    # Validation loop
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
            inputs, targets = [x.to(device) for x in batch]
            labels = targets.view(-1)
            outputs = model(**inputs, labels=labels)
            predictions = torch.argmax(outputs.logits.view(-1, 5), dim=1)
            new_labels, new_predictions = zip(*[(label, prediction) for label, prediction in zip(labels, predictions) if label != -100])
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_preds.extend(torch.tensor(new_predictions).cpu().numpy())
            all_labels.extend(torch.tensor(new_labels).cpu().numpy())

        all_preds_map = [mapping[num] for num in all_preds]
        all_labels_map = [mapping[num] for num in all_labels]

print(accuracy_score([all_preds_map], [all_labels_map]))
print(classification_report([all_preds_map], [all_labels_map]))

model = AutoModelForTokenClassification.from_pretrained('./saved_models')
model.to(device)
test_data = dataset["test"]
test_dataset = CustomDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=coffate_fn)

# Test loop
test_true_labels = []
test_pred_labels = []
wrong_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc=f'Testing'):
        inputs, targets = [x.to(device) for x in batch]
        labels = targets.view(-1)
        outputs = model(**inputs, labels=labels)

        predictions = torch.argmax(outputs.logits.view(-1, 5), dim=1)
        new_labels, new_predictions = zip(*[(label, prediction) for label, prediction in zip(labels, predictions) if label != -100])
        
        test_pred_labels.extend(torch.tensor(new_predictions).cpu().numpy())
        test_true_labels.extend(torch.tensor(new_labels).cpu().numpy())

        if list(new_labels) != list(new_predictions):
            wrong_predictions.append((new_labels, new_predictions))

# Convert predictions and labels to original labels
new_test_predictions = [mapping[num] for num in test_pred_labels]
new_test_labels = [mapping[num] for num in test_true_labels]
confusion_mat = confusion_matrix(new_test_labels, new_test_predictions,labels=["O", "B-Chemical", "I-Chemical", "B-Disease","I-Disease"])
# Print classification report for test set
print(accuracy_score([new_test_predictions], [new_test_labels]))
print(classification_report([new_test_predictions], [new_test_labels]))
print("Confusion Matrix:")
print(confusion_mat)

# 繪製混淆矩陣
plt.figure(figsize=(12, 12))
sns.heatmap(confusion_mat, annot=True, cmap="Blues",cbar_kws={"orientation":"horizontal"}, fmt="d",xticklabels=["O", "B-Chemical", "I-Chemical", "B-Disease","I-Disease"],yticklabels=["O", "B-Chemical", "I-Chemical", "B-Disease","I-Disease"],annot_kws={"size": 16})
# Print classification report for test set
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # 保存成圖片


