# normal
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# pytouch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.optim as optim

# kerasっぽいmodel summaryの表示用
from torchinfo import summary

# cpu, gpuの設定
device = "cpu"

#ファイルの読み込み
pre_data = np.genfromtxt(
    "./csv_data/test.csv",
    skip_header=1,
    usecols=[3, 4, 5, 6, 7],
    delimiter=","  
)

data = pre_data[ pre_data[:, 0] != 0 ][:, :-1]
label = pre_data[ pre_data[:, 0] != 0 ][:, -1]

# 学習データと検証データに分割
x_train, x_valid, t_train, t_valid = train_test_split(data, label, shuffle=True, test_size=0.3)

# Tensor型に変換, GPUで使えるっぽいのでnp.array->tensorにできるんならしたらいいと思う
x_train = torch.from_numpy(x_train).float().to(device)
t_train = torch.from_numpy(t_train).long().to(device)
x_valid = torch.from_numpy(x_valid).float().to(device)
t_valid = torch.from_numpy(t_valid).long().to(device)

# dataset の作成
train_dataset = TensorDataset(x_train, t_train)
valid_dataset = TensorDataset(x_valid, t_valid)

# dataloader の作成（ミニバッチ処理のため）
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# modelの作成とその中身確認
class DNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # return nn.LogSoftmax(dim=1)(x)
        return x

model = DNNmodel().to(device)
summary(model, input_size=(8, 4))

# 損失関数の定義
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# エポック数
num_epochs = 50

# 学習時と検証時で分けるためディクショナリを用意
dataloaders_dict = {
    'train': train_dataloader,
    'val'  : valid_dataloader
}

def train_model(model, train_loader, loss_function, optimizer):
    train_loss = 0.0
    train_acc  = 0.0
    num_train  = 0
    model.train() # train mode
    for inputs, labels in train_loader:
        num_train += len(labels) # count batch number
        optimizer.zero_grad() # initialize grad, ここで初期化しないと過去の重みがそのまま足される
        #1 forward
        outputs = model(inputs)
        #2 calculate loss
        loss = loss_function(outputs, labels)
        # ラベルを予測
        _, preds = torch.max(outputs, dim = 1)
        #3 calculate grad, ここで勾配を計算
        loss.backward()
        #4 update parameters, optimizerに従って勾配をもとに重みづけ
        optimizer.step()
        # 損失関数と正答率を計算して足し上げ
        train_loss += loss.item() * inputs.size(0)
        train_acc  += torch.sum(preds == labels)
    train_loss = train_loss / num_train
    train_acc  = train_acc  / num_train
    return train_loss, train_acc

def valid_model(model, valid_loader, loss_function):
    valid_loss = 0.0
    valid_acc  = 0.0
    num_valid  = 0
    model.eval() # eval mode
    with torch.no_grad(): # invalidate grad
        for inputs, labels in valid_loader:
            num_valid += len(labels)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            _, preds = torch.max(outputs, dim = 1)
            valid_loss += loss.item() * inputs.size(0)
            valid_acc  += torch.sum(preds == labels)
        valid_loss = valid_loss / num_valid
        valid_acc  = valid_acc  / num_valid
    return valid_loss, valid_acc

def learning(model, train_loader, valid_loader, loss_function, optimizer, n_epoch):
    train_loss_list = []
    train_acc_list  = []
    valid_loss_list = []
    valid_acc_list  = []
    # epoch loop
    for epoch in range(n_epoch):
        train_loss, train_acc = train_model(model, train_loader, loss_function, optimizer)
        valid_loss, valid_acc = valid_model(model, valid_loader, loss_function)
        print(f'epoch : {epoch+1:>4}/{n_epoch}, train_loss : {train_loss:.5f}, train_acc : {train_acc:.5f}, valid_loss : {valid_loss:.5f}, valid_acc : {valid_acc:.5f}')
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)        
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
    dict_data = {
        "train": { "loss" : train_loss_list, "acc" : train_acc_list },
        "valid": { "loss" : valid_loss_list, "acc" : valid_acc_list },
    }
    return dict_data

learning( model, train_dataloader, valid_dataloader, criterion, optimizer, 50 )