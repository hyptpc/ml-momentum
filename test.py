import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n_edge = 3
track = np.array([
    [34.4368,	-8.66291,	-98.2076],
    [34.5747,	-9.12695,	-91.2346],
    [34.5767,	-9.13491,	-91.1158],
    [34.7328,	-9.86489,	-80.2171],
    [34.7339,	-9.87252,	-80.1034],
    [34.8096,	-10.5765,	-69.5844],
    [34.8096,	-10.5839,	-69.4737],
    [34.8115,	-11.2676,	-59.1923],
    [34.8111,	-11.2748,	-59.0839],
    [34.7456,	-11.9464,	-48.9638],
])


edge_from = []
edge_to   = []
for i, pos in enumerate(track):
    distance = np.linalg.norm( track - np.full_like(track, pos), axis = 1)
    print(distance)
    edge_from += [i]*n_edge
    edge_to   += np.argsort(distance)[1:n_edge+1].tolist()
edge_index = torch.tensor([edge_from, edge_to])
a = Data(x=torch.tensor(track), y=None, edge_index=edge_index)

# エッジ情報(ノード間の接続を表現)
edge_from = [0, 1, 1, ]
edge_to = [1, 3, 2]
edge_index = torch.tensor([edge_from, edge_to])

# 特徴量Xを設定（ノードごと）
x_0 = [0, 1, 2]
x_1 = [1, 2, 3]
x_2 = [2, 3, 4]
x_3 = [3, 4, 5]
x = torch.tensor([x_0, x_1, x_2, x_3])

# ラベルyを設定（ノードごと）
y_0 = [0]
y_1 = [0]
y_2 = [1]
y_3 = [0]
y = torch.tensor([y_0, y_1, y_2, y_3])

# グラフオブジェクトへの変換
data = Data(x=x, y=y, edge_index=edge_index)

# グラフの可視化
nxg = to_networkx(a)
nx.draw(nxg,
        with_labels = True,
        # node_color = y,
        alpha=0.5)
plt.show()

