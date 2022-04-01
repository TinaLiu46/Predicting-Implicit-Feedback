import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# try linear model to predict 'times'


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)  # .cuda()
        self.item_emb = nn.Embedding(num_items, emb_size)  # .cuda()
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        # initlializing weights
        self.user_emb.weight.data.uniform_(0, 0.05)  # .cuda()
        self.item_emb.weight.data.uniform_(0, 0.05)  # .cuda()
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)  # .cuda()
        V = self.item_emb(v)  # .cuda()

        b = self.user_bias(u).squeeze()
        c = self.item_bias(v).squeeze()
        # return (U*V).sum(1)
        return (U*V).sum(1)+b+c


def valid_loss(model, valid_df):
    model.eval()
    users = torch.LongTensor(valid_df['user_id'].values)  # .cuda()
    items = torch.LongTensor(valid_df['item_id'].values)  # .cuda()
    ratings = torch.FloatTensor(valid_df['binary_ind'].values)  # .cuda()
    y_hat = model(users, items)
    loss = F.binary_cross_entropy(torch.sigmoid(y_hat), ratings)
    return loss.item()

# here we are not using data loaders because our data fits well in memory


def train_epocs(model, train_df, valid_df, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_losses = []
    valid_losses = []
    for i in range(epochs):
        model.train()
        users = torch.LongTensor(train_df.user_id.values)  # .cuda()
        items = torch.LongTensor(train_df.item_id.values)  # .cuda()
        ratings = torch.FloatTensor(train_df.binary_ind.values)  # .cuda()

        y_hat = model(users, items)
        loss = F.binary_cross_entropy(torch.sigmoid(y_hat), ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if valid_df is not None:
            testloss = valid_loss(model, valid_df)
            print("valid loss %.3f" % testloss)
            #print("train loss %.3f valid loss %.3f" % (loss.item(), testloss))
            valid_losses.append(testloss)
        print("train loss %.3f" % loss.item())
        train_losses.append(loss.item())

    if valid_df is not None:
        sns.lineplot(x=range(1, epochs+1), y=valid_losses, label='valid loss')
    sns.lineplot(x=range(1, epochs+1), y=train_losses, label='train loss')
    plt.legend(loc=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
