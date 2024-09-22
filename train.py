# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/22 17:04
# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels):
        self.tokenizer = BertTokenizer.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.sentences = sentences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        label = self.labels[index]
        encoded_pair = self.tokenizer(sent, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        return token_ids, attn_masks, token_type_ids, label

class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        self.bert = BertModel.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.linear = nn.Linear(768, 14)  # 根据类别数量调整
        self.dropout = nn.Dropout(0.5)

    def forward(self, token_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids=token_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('D:/pytorch/nlp_new/1split_1.csv', sep='\t')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    print("Unique labels:", pd.unique(y_train))
    #print("Number of classes:", num_classes)

    train_dataset = MyDataset(X_train.tolist(), y_train.tolist())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    model = BertClassify().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):
        print(f'Epoch {epoch + 1}')
        for token_ids, attn_masks, token_type_ids, labels in train_loader:
            token_ids, attn_masks, token_type_ids, labels = token_ids.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(token_ids, attn_masks, token_type_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'bert_model.pth')

if __name__ == '__main__':
    main()


"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# Dataset preparation
class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None):
        # 确保指定的本地路径没有尾部斜线，且目录正确
        self.tokenizer = BertTokenizer.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        self.sentences = sentences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        label = self.labels[index]
        encoded_pair = self.tokenizer(sent, padding='max_length', truncation=True, max_length=8, return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        return token_ids, attn_masks, token_type_ids, label

# Model Definition
class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        self.bert = BertModel.from_pretrained('D:/pytorch/nlp_new/bert-base-chinese')
        #self.linear = nn.Linear(768, 2)  # Adjust number of output classes if necessary
        self.linear = nn.Linear(768, 14)  # Adjust number of output classes if necessary
        self.dropout = nn.Dropout(0.5)

    def forward(self, token_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids=token_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits

# Main function to train model
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # Load data
    df = pd.read_csv('D:/pytorch/nlp_new/demo.csv', sep='\t')

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    train_dataset = MyDataset(X_train.tolist(), y_train.tolist())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    model = BertClassify().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(10):  # number of epochs
        for token_ids, attn_masks, token_type_ids, labels in train_loader:
            token_ids, attn_masks, token_type_ids, labels = token_ids.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(token_ids, attn_masks, token_type_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'bert_model.pth')

if __name__ == '__main__':
    main()
"""