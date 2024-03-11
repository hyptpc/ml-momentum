# pytouch
from torch import nn
from torch.nn import functional as F

# torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

# modelの作成
class GNNmodel(nn.Module):
    def __init__(self, input_dim, output_dim):
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
