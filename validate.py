# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/22 20:45
# -*- coding: utf-8 -*-
# @Author : Maxiaoxuan
# @Time : 2024/5/22
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.tokenizer = BertTokenizer.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.sentences = sentences
        self.labels = torch.tensor(labels, dtype=torch.long) if with_labels else None
        self.with_labels = with_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        encoded_pair = self.tokenizer(sent, padding='max_length', truncation=True, max_length=8, return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        if self.with_labels:
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        self.bert = BertModel.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.linear = nn.Linear(768, 14)  # Adjust the number of output classes if necessary
        self.dropout = nn.Dropout(0.5)

    def forward(self, token_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids=token_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertClassify().to(device)
    model.load_state_dict(torch.load('bert_model.pth'))
    model.eval()

    df = pd.read_csv('D:/pytorch/nlp_new/1split_2.csv', delimiter='\t')
    X_val, y_val = df['text'].tolist(), df['label'].tolist()

    validate_dataset = MyDataset(X_val, y_val)
    validate_loader = Data.DataLoader(dataset=validate_dataset, batch_size=2, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for token_ids, attn_masks, token_type_ids, labels in validate_loader:
            token_ids, attn_masks, token_type_ids, labels = token_ids.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            outputs = model(token_ids, attn_masks, token_type_ids)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the validation texts: {100 * correct / total}%')
if __name__ == '__main__':
    main()
