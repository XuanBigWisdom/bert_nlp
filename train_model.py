# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/23 9:44
# -*- coding: utf-8 -*-
# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from xuan_tools import XuanDataset, Bert_c

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('D:/pytorch/nlp_new/1split_1.csv', sep='\t')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    train_dataset = XuanDataset(X_train.tolist(), y_train.tolist())
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    model = Bert_c(num_classes=14).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for epoch in range(5):
        total_loss = 0
        for token_ids, attn_masks, token_type_ids, labels in train_loader:
            token_ids, attn_masks, token_type_ids, labels = token_ids.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(token_ids, attn_masks, token_type_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')

    torch.save(model.state_dict(), 'bert_model.pth')
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')

if __name__ == '__main__':
    main()
