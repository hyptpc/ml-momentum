# normal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import sys
import lmfit as lf
import lmfit.models as lfm

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

# original module
import original_module as mod


# cpu, gpuの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの重みデータのパス
# model_path = "model/20240306-003918/model_0094.pt"
# model_path = "model/20240306-090429/model_0197.pt"
model_path = "model/20240307-214232/model_0149.pt"

# input, output sizeの設定
input_dim  = 3  # num of edge feature (energy deposit)
output_dim = 1  # num of output size  (momentum)

# テストデータを読み込んでデータローダー作成
test7208 = mod.DataManager("./csv_data/test7208_EM0.csv")
test = test7208.load_data()
batch_size = 64
num_workers=8
test_dataloader = DataLoader(test, batch_size=batch_size, num_workers=num_workers)

# modelの作成
class GNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 100)
        self.conv2 = GCNConv(100, 200)
        self.linear1 = nn.Linear(200, 100)
        self.linear2 = nn.Linear(100, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.silu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = F.silu(x)
        x = global_mean_pool(x, data.batch)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x.squeeze()
model = GNNmodel().to(device)

# modelの重みの読み込み
model_from_weight = GNNmodel().to(device)
model_from_weight.load_state_dict(torch.load(model_path, map_location=device))

# modelを使って運動量を予想
pred_mom = torch.tensor([]).to(device)
mom = torch.tensor([]).to(device)

model_from_weight.eval() # eval mode
with torch.no_grad(): # invalidate grad
    for inputs, labels in test_dataloader:
        outputs = model_from_weight(inputs.to(device=device))
        pred_mom = torch.cat((pred_mom, outputs))
        mom = torch.cat((mom, labels.to(device)))

# matplotlibで使うためにgpu->cpuに変換
mom = mom.cpu()
pred_mom = pred_mom.cpu()

plt.rcParams['font.size'] = 18
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(mom, pred_mom, "o")
ax1.plot( [mom.min(), mom.max()], [mom.min(), mom.max()], "--", color = "C3" )
ax1.set_xlabel("proton momentum [MeV/c]")
ax1.set_ylabel("predicted momentum [MeV/c]")

hist_data = ax2.hist((mom - pred_mom)/mom*100, bins = np.linspace(-2.5, 2.5, 101))
# hist_data = ax2.hist((mom - pred_mom)/mom*100, bins = np.linspace(-7.5, 7.5, 101))

# hist_data = ax2.hist((mom - pred_mom)/mom*100, bins = np.linspace(-0.5, 0.5, 51))

width = hist_data[1][1]-hist_data[1][0]
fit_x = hist_data[1][:-1] + np.full_like(hist_data[1][:-1], width/2)
fit_y = hist_data[0]
model = lfm.GaussianModel()
params = model.guess( x = fit_x, data = fit_y )
result = model.fit(x = fit_x, data = fit_y, params=params, method='leastsq')
print(result.fit_report())
result_x = np.linspace(np.min(fit_x), np.max(fit_x), 10000)
# fit_y = result.eval_components(x=fit_x)["exponential"]
result_y = result.eval_components(x=result_x)["gaussian"]
ax2.plot(result_x, result_y, c="C1")
ax2.set_xlabel(r"$\Delta p/p$ [%]")
plt.show()