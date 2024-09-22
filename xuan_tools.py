# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/23 9:44
# -*- coding: utf-8 -*-
# xuan_tools.py

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch.nn as nn

class XuanDataset(Dataset):
    def __init__(self, sentences, labels, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.sentences = sentences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        label = self.labels[index]
        encoded_pair = self.tokenizer(sent, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        return token_ids, attn_masks, token_type_ids, label

class Bert_c(nn.Module):
    def __init__(self, num_classes=14):
        super(Bert_c, self).__init__()
        self.bert = BertModel.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, token_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids=token_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits
