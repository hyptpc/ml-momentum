# normal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import sys
import random
import time
from discordwebhook import Discord
discord = Discord(url="https://discord.com/api/webhooks/1195721561018220544/g8IqwcoYDotpcmXTrEgB3o-Bnc_GXsAU_lZzAm7zBZtWi_pjLi3z0LAAtbXxgoI48Tfa")

# pytouch
import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
import torch.optim as optim

# torch_geometric
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, summary

# cpu, gpuの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataManager():
    def __init__(self, path):
        self.path = path
        self.data = np.genfromtxt(
            path,
            skip_header=1,
            delimiter=","
        )

    def convert_graph_data(self, pos_data, features, n_edge = 2): # should be len(pos_data) > n_egde+1
        edge_from = []
        edge_to   = []
        edge_attr = []
        for i, point in enumerate(pos_data):
            distance = np.linalg.norm( pos_data - np.full_like(pos_data, point), axis = 1) # calc. Euclidean distance
            edge_from += [i]*n_edge
            edge_to   += np.argsort(distance)[1:n_edge+1].tolist()
            edge_attr += distance[ np.argsort(distance)[1:n_edge+1] ].tolist()
        edge_index = torch.tensor([edge_from, edge_to])
        return Data(x=torch.tensor(features), y=None, edge_index=edge_index, edge_attr=edge_attr)

    def load_data(self):
        index = 0
        pos_data = []
        features = []
        mom      = []
        data     = []
        for i in tqdm(range(len(self.data))):
            if self.data[i][0] == index:
                pos_data.append([self.data[i][1], self.data[i][2], self.data[i][3]])
                features.append([self.data[i][9]])
                # features.append([self.data[i][1], self.data[i][2], self.data[i][3], self.data[i][4]])
                # features.append([self.data[i][5], self.data[i][6], self.data[i][7]])
                mom.append(self.data[i][5])
            else:
                if len(pos_data) > 5:
                    data.append([ 
                        self.convert_graph_data(np.array(pos_data), features), 
                        statistics.mean(mom)
                    ])
                index += 1
                pos_data = [[self.data[i][1], self.data[i][2], self.data[i][3]]]
                features = [[self.data[i][9]]]
                # features = [[self.data[i][1], self.data[i][2], self.data[i][3], self.data[i][4]]]
                # features = [[self.data[i][5], self.data[i][6], self.data[i][7]]]
                mom = [self.data[i][5]]

        return data

def shuffle_list_data(data, ratio = 0.2):
    n_data = len(data)
    n_valid_data = int( n_data*ratio )
    shuffled_data = random.sample(data, n_data)
    return shuffled_data[n_valid_data:], shuffled_data[:n_valid_data]

gen7208 = DataManager("./csv_data/gen7208.csv")
data = gen7208.load_data()

# 学習データと検証データに分割
train_data, valid_data = shuffle_list_data(data)

# dataloader の作成（ミニバッチ処理のため）
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=8, shuffle=False)

# modelの作成とその中身確認
class GNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 100)
        self.conv2 = GCNConv(100, 200)
        self.linear1 = nn.Linear(200,100)
        self.linear2 = nn.Linear(100,1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = self.conv1(x, edge_index)
        x = F.silu(x)
        x = self.conv2(x, edge_index)
        x = F.silu(x)
        x = global_mean_pool(x, data.batch)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x.squeeze()

model = GNNmodel().to(device)

# model_from_weight = GNNmodel()
# model_from_weight.load_state_dict(torch.load('model.pt', map_location=device))

# test7208 = DataManager("./csv_data/test7208.csv")
# test = test7208.load_data()
# test_dataloader = DataLoader(test, num_workers=8)

# model.eval() # eval mode
# with torch.no_grad(): # invalidate grad
#     for inputs, labels in test_dataloader:
#         inputs.to(device=device)    
#         outputs = model(inputs)
#         print(outputs, labels)

# sys.exit()

# 損失関数の定義
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters())

# エポック数
num_epochs = 100

# 学習時と検証時で分けるためディクショナリを用意
dataloaders_dict = {
    'train': train_dataloader,
    'val'  : valid_dataloader
}

def train_model(model, train_loader, loss_function, optimizer):
    train_loss = 0.0
    num_train  = 0
    model.train() # train mode
    for inputs, labels in train_loader:
        inputs.to(device=device)
        num_train += len(labels) # count batch number
        optimizer.zero_grad() # initialize grad, ここで初期化しないと過去の重みがそのまま足される
        #1 forward
        outputs = model(inputs)
        #2 calculate loss
        loss = loss_function(outputs, labels.to(device=device))
        #3 calculate grad, ここで勾配を計算
        loss.backward()
        #4 update parameters, optimizerに従って勾配をもとに重みづけ
        optimizer.step()
        # 損失関数と正答率を計算して足し上げ
        train_loss += loss.item() * inputs.size(0)
    train_loss = train_loss / num_train
    return train_loss

def valid_model(model, valid_loader, loss_function):
    valid_loss = 0.0
    num_valid  = 0
    model.eval() # eval mode
    with torch.no_grad(): # invalidate grad
        for inputs, labels in valid_loader:
            inputs.to(device=device)
            num_valid += len(labels)
            outputs = model(inputs)
            loss = loss_function(outputs, labels.to(device=device))
            valid_loss += loss.item() * inputs.size(0)
        valid_loss = valid_loss / num_valid
    return valid_loss

def learning(model, train_loader, valid_loader, loss_function, optimizer, n_epoch):
    train_loss_list = []
    train_time_list = []
    valid_loss_list = []
    # epoch loop
    for epoch in range(n_epoch):
        start = time.time()
        train_loss = train_model(model, train_loader, loss_function, optimizer)
        end   = time.time()
        valid_loss = valid_model(model, valid_loader, loss_function)
        print(f'epoch : {epoch+1:>4}/{n_epoch}, train_loss : {train_loss:.3f}, valid_loss : {valid_loss:.3f}, time : {end-start:.3f}')
        train_loss_list.append(train_loss)
        train_time_list.append(end-start)
        valid_loss_list.append(valid_loss)
    dict_data = {
        "train": { "loss" : train_loss_list, "time" : train_time_list },
        "valid": { "loss" : valid_loss_list },
    }
    return dict_data

dict_data = learning( model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs )
torch.save(model.state_dict(), 'model_pos.pt')

# import csv
# tmp1 = np.array(dict_data["train"]["loss"])
# tmp2 = np.array(dict_data["valid"]["loss"])
# tmp3 = np.array(dict_data["train"]["time"])
# buf = np.vstack([tmp1, tmp2, tmp3]).T
# with open("without_apex.csv", mode = "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["train_loss", "valid_loss", "train_time"])
#     for line in buf:
#         writer.writerow(line)

discord.post(content="finish estimate_mom.py")

# plt.plot(dict_data["train"]["loss"])
# plt.plot(dict_data["valid"]["loss"])
# plt.show()
    