# normal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import sys
import os
import random
import time
import datetime
import csv

# pytouch
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

# torch_geometric
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, summary

# original module
import original_module as mod


# cpu, gpuの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input, output sizeの設定
input_dim  = 3 # num of edge feature (energy deposit)
output_dim = 1  # num of output size  (momentum)

# エポック数
num_epochs = 50

# データ読み込み
gen7208 = mod.DataManager("./csv_data/gen7208_EM0_1.csv")
data = gen7208.load_data(isDebug=0)

# 学習データと検証データに分割
train_data, valid_data = mod.shuffle_list_data(data)

# dataloader の作成（ミニバッチ処理のため）
batch_size  = 64
num_workers = 8
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# modelの作成
class GNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, 256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.silu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = F.silu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = F.silu(x)
        x = global_mean_pool(x, data.batch)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        x = F.silu(x)
        x = self.linear3(x)
        return x.squeeze()
model = GNNmodel().to(device)

# 損失関数などの定義
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5, patience=3, min_lr=0.0001)

# 学習時と検証時で分けるためディクショナリを用意
dataloaders_dict = {
    'train': train_dataloader,
    'val'  : valid_dataloader
}

# modelなどの読み込み
checkpoint = torch.load("model/20240312-012520/checkpoint.bin") 
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])

dict_data = mod.learning( device, model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, scheduler )

# discordで通知( https://pypi.org/project/discordwebhook/ )
try:
    from discordwebhook import Discord
    import discord_url  # 自作関数でurlを単に格納しているだけ
    discord = discord_url.get_discord()
    discord.post(content="({}) finish estimate_mom.py".format(sys.platform))
except:
    pass

plt.plot(dict_data["train"]["loss"])
plt.plot(dict_data["valid"]["loss"])
plt.show()
