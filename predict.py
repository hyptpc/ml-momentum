# normal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import sys
import random
import time

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
                # features.append([self.data[i][2], self.data[i][4]])
                features.append([self.data[i][5], self.data[i][6], self.data[i][7]])
                mom.append(self.data[i][5])
            else:
                if len(pos_data) > 5:
                    data.append([ 
                        self.convert_graph_data(np.array(pos_data), features), 
                        statistics.mean(mom)
                    ])
                index += 1
                pos_data = [[self.data[i][1], self.data[i][2], self.data[i][3]]]
                # features = [[self.data[i][2], self.data[i][4]]]
                features = [[self.data[i][5], self.data[i][6], self.data[i][7]]]
                mom = [self.data[i][5]]

        return data


# modelの作成とその中身確認
class GNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 100)
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

model_from_weight = GNNmodel().to(device)
model_from_weight.load_state_dict(torch.load('model_wap.pt', map_location=device))

test7208 = DataManager("./csv_data/test7208.csv")
test = test7208.load_data()
batch_size = 64
test_dataloader = DataLoader(test, batch_size=batch_size, num_workers=8)

pred_mom = torch.tensor([]).to(device)
mom = torch.tensor([]).to(device)

model_from_weight.eval() # eval mode
with torch.no_grad(): # invalidate grad
    for inputs, labels in test_dataloader:
        inputs.to(device=device)
        outputs = model_from_weight(inputs)
        # print(outputs.item(), labels.item())
        pred_mom = torch.cat((pred_mom, outputs))
        mom = torch.cat((mom, labels.to(device)))
        # mom.append([outputs.item(), labels.item()])
        print( pred_mom )

mom = np.array(mom)
plt.plot(mom[:, 0], mom[:, 1], "o")
plt.show()

plt.hist((mom[:, 0] - mom[:, 1])/mom[:, 1], bins = 100)
plt.show()