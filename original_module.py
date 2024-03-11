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

class DataManager():
    def __init__(self, path):
        self.path = path
        self.data = np.genfromtxt(
            path,
            skip_header=1,
            delimiter=","
        )

    def convert_graph_data(self, pos_data, n_edge = 2): # should be len(pos_data) > n_egde+1
        edge_from = []
        edge_to   = []
        edge_attr = []
        features  = []
        past_pos  = pos_data[0]
        for i, point in enumerate(pos_data):
            distance = np.linalg.norm( pos_data - np.full_like(pos_data, point), axis = 1) # calc. Euclidean distance
            edge_from += [i]*n_edge
            edge_to   += np.argsort(distance)[1:n_edge+1].tolist()
            edge_attr += distance[ np.argsort(distance)[1:n_edge+1] ].tolist()
            diff = pos_data[i] - past_pos
            dis = np.linalg.norm( diff )
            if (dis != 0):
                features.append([ diff[0]/dis, diff[1]/dis, diff[2]/dis ])
            else:
                features.append( [0., 0., 0.] )
            past_pos = pos_data[i]
        graph_data = Data(
            x          = torch.tensor(features),             # node feature
            y          = None,                               # node label
            edge_index = torch.tensor([edge_from, edge_to]), # edge
            edge_attr  = torch.tensor(edge_attr)             # edge feature
        )
        return graph_data

    def load_data(self, isDebug = False):
        evnum = 0
        pos_data = []
        mom      = []
        dataset  = []
        for i in tqdm(range(len(self.data))):
            if self.data[i][0] == evnum:
                pos_data.append([self.data[i][1], self.data[i][2], self.data[i][3]]) # [x, y, z]
                mom.append(self.data[i][5]) # momentum
            else:
                if len(pos_data) > 5:
                    graph_data = self.convert_graph_data(np.array(pos_data), 2)
                    dataset.append([ 
                        graph_data,
                        mom[0]
                    ])
                    # --------------------------------------
                    # draw graph data
                    # --------------------------------------
                    if isDebug:
                        plt.rcParams['font.size'] = 18
                        nxg = to_networkx(graph_data)
                        fig = plt.figure(figsize=(10, 6))
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122)
                        for j, pos in enumerate(pos_data):
                            ax1.scatter( pos[0], pos[2], color = "k", marker="${}$".format(j))
                        ax1.set_xlabel("X")
                        ax1.set_ylabel("Z")
                        nx.draw(nxg,
                                node_color = 'w',
                                edgecolors = 'k', # node border color
                                with_labels = True,
                                edge_color = graph_data.edge_attr.mul(1/graph_data.edge_attr.max()).tolist(),
                                edge_cmap = plt.cm.plasma,
                                alpha = 0.5, ax=ax2)
                        plt.show()
                    # --------------------------------------
                pos_data = [[self.data[i][1], self.data[i][2], self.data[i][3]]] # [x, y, z]
                mom = [self.data[i][5]] # momentum
                evnum += 1
        return dataset

def shuffle_list_data(data, ratio = 0.2):
    # shuffule and divide dataset
    n_data = len(data)
    n_valid_data = int( n_data*ratio )
    shuffled_data = random.sample(data, n_data)
    return shuffled_data[n_valid_data:], shuffled_data[:n_valid_data]

def save_history(dict_data, save_path):
    train_loss = np.array(dict_data["train"]["loss"])
    valid_loss = np.array(dict_data["valid"]["loss"])
    train_time = np.array(dict_data["train"]["time"])
    buf = np.vstack([train_loss, valid_loss, train_time]).T
    with open("{}/history.csv".format(save_path), mode = "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_loss", "valid_loss", "train_time"])
        for line in buf:
            writer.writerow(line)

def train_model(model, train_loader, loss_function, optimizer, device):
    train_loss = 0.0
    num_train  = 0
    model.train() # train mode
    for inputs, labels in train_loader:
        num_train += len(labels) # count batch number
        optimizer.zero_grad() # initialize grad, ここで初期化しないと過去の重みがそのまま足される
        #1 forward
        outputs = model(inputs.to(device=device))
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

def valid_model(model, valid_loader, loss_function, device):
    valid_loss = 0.0
    num_valid  = 0
    model.eval() # eval mode
    with torch.no_grad(): # invalidate grad
        for inputs, labels in valid_loader:
            num_valid += len(labels)
            outputs = model(inputs.to(device=device))
            loss = loss_function(outputs, labels.to(device=device))
            valid_loss += loss.item() * inputs.size(0)
        valid_loss = valid_loss / num_valid
    return valid_loss

def learning(device, model, train_loader, valid_loader, loss_function, optimizer, n_epoch, scheduler = None):
    now = datetime.datetime.now()
    save_path = "./model/{}{:0=2}{:0=2}-{:0=2}{:0=2}{:0=2}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs(save_path, exist_ok=True)
    train_loss_list = []
    train_time_list = []
    valid_loss_list = []
    valid_loss_min  = np.inf
    # epoch loop
    for epoch in range(n_epoch):
        start = time.time()
        train_loss = train_model(model, train_loader, loss_function, optimizer, device)
        end   = time.time()
        valid_loss = valid_model(model, valid_loader, loss_function, device)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch : {epoch+1:>4}/{n_epoch}, train_loss : {train_loss:.3f}, valid_loss : {valid_loss:.3f}, time : {end-start:.3f}, lr : {lr}')
        train_loss_list.append(train_loss)
        train_time_list.append(end-start)
        valid_loss_list.append(valid_loss)
        if (valid_loss < valid_loss_min):
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), '{}/model_{:0=4}.pt'.format(save_path, epoch))
        if (scheduler != None):
            scheduler.step(valid_loss)
    dict_data = {
        "train": { "loss" : train_loss_list, "time" : train_time_list },
        "valid": { "loss" : valid_loss_list },
    }
    save_history(dict_data, save_path)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, "{}/checkpoint.bin".format(save_path))

    return dict_data