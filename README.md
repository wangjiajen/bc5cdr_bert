# <font color="#0000AA"> **Named Entity Recognition using DistilBERT**</font>
<br>`本專案旨在展示使用DistilBERT架構訓練Named Entity Recognition (NER)模型的代碼。該模型在BC5CDR數據集上進行訓練，該數據集包含了帶有生物醫學註釋的文檔。`<br>

## 環境設置

<br>- Python 版本: 3.8
<br>- 所需套件: 
* Python 3.x
* PyTorch
* Transformers library (Hugging Face)
* Pandas
* NumPy
* tqdm
* seqeval

```sh
# Clone
git clone https://github.com/wangjiajen/bc5cdr_bert.git

# Install dependencies.
pipenv install
```
## 準備數據

```sh
# 下載數據集
dataset = load_dataset("tner/bc5cdr")

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]
```
## 訓練過程

<br>代碼概述
- `bc5cdr_bert.py`：主要的訓練程式碼，包含加載數據、定義模型架構、訓練模型以及在驗證集和測試集上評估模型的代碼。

- `CustomDataset`：用於創建自定義的PyTorch數據集，用於加載BC5CDR數據。

- `coffate_fn`：此函數用作DataLoader的聚合函數。它對輸入文本進行標記並處理訓練標籤。

- `mapping`：此字典將數值標籤映射到其對應的NER標籤。
```sh
mapping = {
    0: 'O',
    1: "B-Chemical",
    2: "B-Disease",
    3: "I-Disease",
    4: "I-Chemical"
}
```

<br>在訓練之前，超參數的調整能獲得更好的模型。例如：
- `LR`：學習率 (Learning Rate)。
- `EPOCHS`：影響的就是迭代計算的次數。
- `BATCH_SIZE`：影響的是訓練過程中的完成每個epoch所需的時間和每次迭代(iteration)之間梯度的平滑程度。<br>

## 評估和測試

<br>在訓練過程中，模型的性能需要在驗證集上進行評估，以確保它能夠泛化到未見過的數據。這些評估指標可以幫助您了解模型的準確性和效能。在這個項目中，我們使用了以下評估指標：
* `準確性 (Accuracy)`: 正確預測的樣本數與總樣本數的比例。可以使用 accuracy_score 函數來計算。

* `分類報告 (Classification Report)`: 包括了每個類別的準確率、召回率和 F1 分數等評估指標。可以使用 classification_report 函數來生成。

## 結果


|                   |Train   |Test    |
|     :--------:    |--------|--------|
|Accuracy           |95.23%  |95.12%  |
|Precision          |0.89    |0.89    |
|Recall             |0.89    |0.87    |
|F1-score           |0.89    |0.88    |

## 參考資料

<br>[1]:https://huggingface.co/datasets/tner/bc5cdr?fbclid=IwAR06QVaGqi1IvVCjYKudvo2Agae_xsDRmdgFIkDktQUUePIJiIX5oBKyujs<br>
[2]:https://huggingface.co/docs/transformers/tasks/token_classification<br>
[3]:https://blog.csdn.net/jclian91/article/details/104785226<br>


