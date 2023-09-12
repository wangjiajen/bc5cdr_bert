#In[]
import pandas as pd

#In[]
df_train=pd.read_json('/home/apple/bc5cdr_bert/dataset/train.json',lines=True)
df_test=pd.read_json('/home/apple/bc5cdr_bert/dataset/test.json',lines=True)
df_valid=pd.read_json('/home/apple/bc5cdr_bert/dataset/valid.json',lines=True)

tokens_list = df_train['tokens']
ner_tags_list=df_train['tags']
test_tokens_list = df_test['tokens']
test_ner_tags_list=df_test['tags']
valid_tokens_list = df_valid['tokens']
valid_ner_tags_list=df_valid['tags']

#In[]
def list_to_dataframe(tokens_list, ner_tags_list):
  df = pd.DataFrame(zip(tokens_list,ner_tags_list), columns = ['tokens','tags']) 
  # 至少包含兩個特定的欄位名稱：'tokens' 和 'ner_tags'
  assert all(i for i in ['tokens','tags'] if i in df.columns) 
  # pre-processing
  # 刪除包含 NaN 值的列，移除DataFrame中包含缺失值的行
  df = df.dropna()
  # 刪除 'tokens' 中包含空字符串的行
  df = df.drop(df[df['tokens']==''].index.values, )
  # reset_index() 會將 DataFrame 的索引重新設定為默認的整數索引（0, 1, 2, ...）
  # drop=True 表示在重新設定索引的同時，將原先的索引刪除，否則原先的索引會成為一列新的數據
  df = df.reset_index(drop=True)
  # x.strip() 用於去掉字串兩端的空格
  # x.split(' ') 用於將字串按照空格進行分割，生成一個由單詞組成的清單
  df['tokens'] = df['tokens']
  df['tags'] = df['tags']
  df.head()
  return df

df_train = list_to_dataframe(tokens_list, ner_tags_list)
df_test = list_to_dataframe(test_tokens_list, test_ner_tags_list)
df_valid = list_to_dataframe(valid_tokens_list, valid_ner_tags_list)



#In[]
tag_name = ['O',
            'B-Chemical',
            'B-Disease',
            'I-Disease',
            'I-Chemical']
# len(tag_name)=5 五個類別
from datasets import Dataset, ClassLabel, Sequence, Features, Value, DatasetDict
tags = ClassLabel(num_classes=len(tag_name), names=tag_name)
# 建立字典 dataset_structure：描述資料集的結構的一個定義
# ner_tags是資料集中的一個特徵，而且它的類型是Sequence(序列資料)，且每個項目的類別是tags
dataset_structure = {"ner_tags":Sequence(tags),
                 'tokens': Sequence(feature=Value(dtype='string'))}

#In[]
def df_to_dataset(df, columns=['ner_tags', 'tokens']):
  assert set(['ner_tags', 'tokens']).issubset(df.columns)

  ner_tags = df['ner_tags'].map(tags.str2int).values.tolist()
  tokens = df['tokens'].values.tolist()
  # 確保 tokens 和 ner_tags 中的每個元素都是列表
  assert isinstance(tokens[0], list) 
  assert isinstance(ner_tags[0], list)
  dic = {'ner_tags':ner_tags, 'tokens':tokens}
  # create dataset
  dataset = Dataset.from_dict(mapping=dic,
              features=Features(dataset_structure),)
  return dataset
  
dataset = df_to_dataset(df) # 從train.txt變成df，然後轉成訓練資料dataset
test_dataset =  df_to_dataset(test_df) # 從test-submit.txt變成test_df，然後轉成訓練資料test_dataset


label_names = dataset["train"].features["ner_tags"].feature.names

#In[]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") 

def tokenize_function(examples):
    return tokenizer(examples["tokens"], padding="max_length",
                     truncation=True, is_split_into_words=True)

tokenized_datasets_ = dataset.map(tokenize_function, batched=True)