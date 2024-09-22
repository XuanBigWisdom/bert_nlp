# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/22 20:34
# test.py
# test.py
# This Python script is used to load a trained BERT model and predict labels for a test dataset.

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class BertClassify(nn.Module):
    def __init__(self, bert_path):
        super(BertClassify, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear = nn.Linear(768, 14)  # Adjust based on number of classes in actual training
        self.dropout = nn.Dropout(0.5)

    def forward(self, token_ids, attn_masks):
        outputs = self.bert(input_ids=token_ids, attention_mask=attn_masks)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits

class MyDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=8):
        self.data = pd.read_csv(filepath, delimiter='\t')
        self.texts = self.data['text'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        encoded_pair = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        return token_ids, attn_masks

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for token_ids, attn_masks in data_loader:
            token_ids = token_ids.to(device)
            attn_masks = attn_masks.to(device)
            logits = model(token_ids, attn_masks)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'D:/pytorch/nlp_new/bert/bert_model.pth'
    bert_path = 'D:/pytorch/nlp_new/bert-base-chinese'
    test_file_path = 'D:/pytorch/nlp_new/test_split_1.csv'
    output_file_path = 'D:/pytorch/nlp_new/test_predictions.csv'

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    test_dataset = MyDataset(test_file_path, tokenizer)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

    model = BertClassify(bert_path).to(device)
    model.load_state_dict(torch.load(model_path))
    predictions = predict(model, test_loader, device)

    # Save predictions to CSV
    test_data = pd.read_csv(test_file_path, delimiter='\t')
    test_data['predicted_label'] = predictions
    test_data.to_csv(output_file_path, index=False)

if __name__ == '__main__':
    main()
