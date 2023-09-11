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

tag_name = ['O',
            'B-Chemical',
            'B-Disease',
            'I-Disease',
            'I-Chemical']

#In[]
from datasets import Dataset, ClassLabel, Sequence, Features, Value, DatasetDict
tags = ClassLabel(num_classes=len(tag_name), names=tag_name)
# dataset_structure = {"ner_tags":Sequence(tags),
#                  'tokens': Sequence(feature=Value(dtype='string'))}

# #In[]
# def df_to_dataset(df, columns=['ner_tags', 'tokens']):
#   assert set(['ner_tags', 'tokens']).issubset(df.columns)

#   ner_tags = df['ner_tags'].map(tags.str2int).values.tolist()
#   tokens = df['tokens'].values.tolist()

#   assert isinstance(tokens[0], list) 
#   assert isinstance(ner_tags[0], list)
#   d = {'ner_tags':ner_tags, 'tokens':tokens}# 如果有其他欄位例如id, spans請從這裡添加
#   # create dataset
#   dataset = Dataset.from_dict(mapping=d,
#               features=Features(dataset_structure),)
#   return dataset
  
# dataset = df_to_dataset(df) # 從train.txt變成df，然後轉成訓練資料dataset
# test_dataset =  df_to_dataset(test_df) # 從test-submit.txt變成test_df，然後轉成訓練資料test_dataset

# train = dataset.train_test_split(test_size=0.1) # 訓練做分割，保留validation set

# # Split the 10% test + valid in half test, half valid
# test_valid = train['test'].train_test_split(test_size=0.5)
# # gather everyone if you want to have a single DatasetDict
# dataset = DatasetDict({
#     'train': train['train'], # Trainer會用到
#     'test': test_dataset, # 獨立的測試資料
#     'valid': test_valid['train']}) # Trainer會用到

# label_names = dataset["train"].features["ner_tags"].feature.names

# #In[]
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# def tokenize_function(examples):
#     return tokenizer(examples["tokens"], padding="max_length",
#                      truncation=True, is_split_into_words=True)

# #Get the values for input_ids, attention_mask, adjusted labels
# def tokenize_adjust_labels(all_samples_per_split):
#   tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"],
#                 is_split_into_words=True, truncation=True)
#   # print(tokenized_samples['input_ids'][:2])
#   # tokenizer(string, padding=True, truncation=True) 
#   # assert False
#   print(len(tokenized_samples["input_ids"]))
#   print(tokenized_samples.word_ids(batch_index=2))
#   total_adjusted_labels = []
  
#   for k in range(0, len(tokenized_samples["input_ids"])):
#     prev_wid = -1
#     word_ids_list = tokenized_samples.word_ids(batch_index=k)
#     existing_label_ids = all_samples_per_split["ner_tags"][k]
#     i = -1
#     adjusted_label_ids = []
#     # print(existing_label_ids)
#     # print(adjusted_label_ids)
#     # assert False
#     for word_idx in word_ids_list:
#       # Special tokens have a word id that is None. We set the label to -100 so they are automatically
#       # ignored in the loss function.
#       if(word_idx is None):
#         adjusted_label_ids.append(-100)
#       elif(word_idx!=prev_wid):
#         i = i + 1
#         adjusted_label_ids.append(existing_label_ids[i])
#         prev_wid = word_idx
#       else:
#         label_name = label_names[existing_label_ids[i]]
#         adjusted_label_ids.append(existing_label_ids[i])
        
#     total_adjusted_labels.append(adjusted_label_ids)
  
#   #add adjusted labels to the tokenized samples
#   tokenized_samples["labels"] = total_adjusted_labels
#   return tokenized_samples

# tokenized_dataset = dataset.map(tokenize_adjust_labels,
#                 batched=True,
#                 remove_columns=list(dataset["train"].features.keys()))

# from transformers import DataCollatorForTokenClassification
# data_collator = DataCollatorForTokenClassification(tokenizer)

# #check if gpu is present
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device